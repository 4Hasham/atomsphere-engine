// Bring-your-own-model uploads. Signed-in users only (the API route must use the
// Cognito authorizer, and every request is checked for an owner here).
//
// POST { action: "start", title, category?, size, sizeMeters? }
//   Creates an "uploading" job owned by the caller and returns a presigned POST that
//   only accepts one .glb of at most MAX_MODEL_BYTES, straight to the output bucket.
// POST { action: "complete", jobId }
//   Checks the uploaded object (owner, size, glTF 2.0 binary header) and marks the job
//   "generated", which is what modelUrl.js requires before it will sign a download.
// POST { action: "cancel", jobId }
//   The browser upload failed or was abandoned: delete any partial file, mark failed.
//
// The bucket stays private: the only way to read a model is modelUrl.js, which signs
// a short-lived link for the job's owner only.

import crypto from 'node:crypto';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, PutCommand, QueryCommand, UpdateCommand } from '@aws-sdk/lib-dynamodb';
import { DeleteObjectCommand, GetObjectCommand, HeadObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { createPresignedPost } from '@aws-sdk/s3-presigned-post';

const region = process.env.AWS_REGION || 'us-east-1';
const dynamodb = DynamoDBDocumentClient.from(new DynamoDBClient({ region }));
const s3 = new S3Client({ region });

const MAX_MODEL_BYTES = 200 * 1024 * 1024;
const MAX_TITLE_LENGTH = 80;
const UPLOAD_URL_TTL_SECONDS = 15 * 60;
const ALLOWED_CATEGORIES = new Set([
  'building_structure',
  'mechanical_device',
  'consumer_product',
  'furniture_decor',
  'vehicle',
  'character_creature',
  'nature_landscape',
  'home_appliance',
  'other',
]);

const corsHeaders = {
  'Access-Control-Allow-Origin': process.env.ALLOWED_ORIGIN || 'http://localhost:5173',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
  'Access-Control-Allow-Methods': 'POST,OPTIONS',
  Vary: 'Origin',
};

function response(statusCode, body) {
  return {
    statusCode,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
}

function getOwner(event) {
  const authorizer = event?.requestContext?.authorizer || {};
  const claims = authorizer.jwt?.claims || authorizer.claims || authorizer.lambda || {};
  const userId = claims.sub
    || claims.userId
    || claims.principalId
    || claims['cognito:username']
    || authorizer.principalId;
  if (!userId) return null;
  return {
    userId: String(userId),
    uploadedBy: String(claims.email || claims.preferred_username || claims['cognito:username'] || userId),
  };
}

function getPayload(event) {
  if (event?.body && typeof event.body === 'object') return event.body;
  if (typeof event?.body !== 'string' || !event.body.trim()) return {};
  return JSON.parse(event.isBase64Encoded
    ? Buffer.from(event.body, 'base64').toString('utf8')
    : event.body);
}

function cleanText(value) {
  return typeof value === 'string' ? value.replace(/[\u0000-\u001f\u007f]+/g, ' ').trim() : '';
}

async function findOwnedJob(tableName, jobId, ownerId) {
  const result = await dynamodb.send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: 'job_id = :jobId',
    ExpressionAttributeValues: { ':jobId': jobId },
    ConsistentRead: true,
    Limit: 2,
  }));
  return result.Items?.find((item) => item.owner_id === ownerId) || null;
}

async function startUpload(tableName, bucket, owner, payload) {
  const title = cleanText(payload.title);
  if (!title) return response(400, { error: 'Give the model a name.' });
  if (title.length > MAX_TITLE_LENGTH) {
    return response(400, { error: `The name must be ${MAX_TITLE_LENGTH} characters or fewer.` });
  }
  const category = payload.category ? String(payload.category) : 'other';
  if (!ALLOWED_CATEGORIES.has(category)) return response(400, { error: 'Choose a valid model category.' });
  const size = Number(payload.size);
  if (!Number.isInteger(size) || size < 20 || size > MAX_MODEL_BYTES) {
    return response(400, { error: 'Models must be .glb files of 200 MB or less.' });
  }
  // Largest side in metres, measured in the browser; Studio opens the model at this size.
  const sizeMeters = Number(payload.sizeMeters);

  const jobId = `upload-${crypto.randomUUID()}`;
  const key = `user-models/${jobId}.glb`;
  const now = new Date().toISOString();
  const item = {
    job_id: jobId,
    created_at: now,
    updated_at: now,
    owner_id: owner.userId,
    uploaded_by: owner.uploadedBy,
    model: 'upload',
    input_mode: 'upload',
    category,
    title,
    status: 'uploading',
    output_bucket: bucket,
    output_key: key,
  };
  if (Number.isFinite(sizeMeters) && sizeMeters > 0) item.size_m = Math.round(sizeMeters * 1000) / 1000;

  await dynamodb.send(new PutCommand({
    TableName: tableName,
    Item: item,
    ConditionExpression: 'attribute_not_exists(job_id)',
  }));

  // A presigned POST (not PUT) so S3 itself rejects anything over the size limit.
  const upload = await createPresignedPost(s3, {
    Bucket: bucket,
    Key: key,
    Conditions: [
      ['content-length-range', 20, MAX_MODEL_BYTES],
      ['eq', '$Content-Type', 'model/gltf-binary'],
    ],
    Fields: { 'Content-Type': 'model/gltf-binary' },
    Expires: UPLOAD_URL_TTL_SECONDS,
  });

  return response(200, { jobId, upload });
}

async function completeUpload(tableName, owner, payload) {
  const jobId = payload.jobId || payload.job_id;
  if (typeof jobId !== 'string' || !/^upload-[A-Za-z0-9-]{1,64}$/.test(jobId)) {
    return response(400, { error: 'A valid jobId is required.' });
  }

  const job = await findOwnedJob(tableName, jobId, owner.userId);
  if (!job) return response(404, { error: 'Upload not found.' });
  if (job.status === 'generated') return response(200, { jobId, status: 'generated' });
  if (job.status !== 'uploading') return response(409, { error: 'This upload can no longer be completed.' });

  async function fail(message) {
    await s3.send(new DeleteObjectCommand({ Bucket: job.output_bucket, Key: job.output_key })).catch(() => {});
    await setStatus(tableName, job, 'failed');
    return response(400, { error: message });
  }

  let head;
  try {
    head = await s3.send(new HeadObjectCommand({ Bucket: job.output_bucket, Key: job.output_key }));
  } catch {
    return response(409, { error: 'The file has not finished uploading.' });
  }
  if (!head.ContentLength || head.ContentLength > MAX_MODEL_BYTES) {
    return fail('Models must be .glb files of 200 MB or less.');
  }

  // GLB header: magic "glTF", version 2, then the total length in bytes.
  const start = await s3.send(new GetObjectCommand({
    Bucket: job.output_bucket,
    Key: job.output_key,
    Range: 'bytes=0-11',
  }));
  const header = Buffer.from(await start.Body.transformToByteArray());
  const isGlb = header.length === 12
    && header.toString('ascii', 0, 4) === 'glTF'
    && header.readUInt32LE(4) === 2
    && header.readUInt32LE(8) === head.ContentLength;
  if (!isGlb) return fail("This file isn't a valid glTF 2.0 binary (.glb).");

  await setStatus(tableName, job, 'generated');
  return response(200, { jobId, status: 'generated' });
}

async function cancelUpload(tableName, owner, payload) {
  const jobId = payload.jobId || payload.job_id;
  if (typeof jobId !== 'string' || !/^upload-[A-Za-z0-9-]{1,64}$/.test(jobId)) {
    return response(400, { error: 'A valid jobId is required.' });
  }
  const job = await findOwnedJob(tableName, jobId, owner.userId);
  if (!job) return response(404, { error: 'Upload not found.' });
  if (job.status !== 'uploading') return response(200, { jobId, status: job.status });

  await s3.send(new DeleteObjectCommand({ Bucket: job.output_bucket, Key: job.output_key })).catch(() => {});
  await setStatus(tableName, job, 'failed');
  return response(200, { jobId, status: 'failed' });
}

async function setStatus(tableName, job, status) {
  await dynamodb.send(new UpdateCommand({
    TableName: tableName,
    Key: { job_id: job.job_id, created_at: job.created_at },
    UpdateExpression: 'SET #status = :status, updated_at = :updatedAt',
    ExpressionAttributeNames: { '#status': 'status' },
    ExpressionAttributeValues: { ':status': status, ':updatedAt': new Date().toISOString() },
    ConditionExpression: 'attribute_exists(job_id)',
  }));
}

export const handler = async (event) => {
  const method = event?.httpMethod || event?.requestContext?.http?.method;
  if (method === 'OPTIONS') return { statusCode: 204, headers: corsHeaders, body: '' };
  if (method && method !== 'POST') return response(405, { error: 'Method not allowed.' });

  const tableName = process.env.JOB_STATUS_TABLE;
  const bucket = process.env.OUTPUT_BUCKET;
  if (!tableName || !bucket) {
    return response(500, { error: 'JOB_STATUS_TABLE and OUTPUT_BUCKET environment variables are required.' });
  }

  const owner = getOwner(event);
  if (!owner) return response(401, { error: 'Sign in to upload a model.' });

  let payload;
  try {
    payload = getPayload(event);
  } catch {
    return response(400, { error: 'Request body must be valid JSON.' });
  }

  try {
    if (payload.action === 'start') return await startUpload(tableName, bucket, owner, payload);
    if (payload.action === 'complete') return await completeUpload(tableName, owner, payload);
    if (payload.action === 'cancel') return await cancelUpload(tableName, owner, payload);
    return response(400, { error: 'action must be "start", "complete" or "cancel".' });
  } catch (error) {
    console.error('Model upload request failed:', error);
    return response(500, { error: 'The upload could not be processed.' });
  }
};
