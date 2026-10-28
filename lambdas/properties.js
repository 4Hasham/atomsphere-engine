// Properties (floorplan → 3D house). Signed-in users only (the API route must use the
// Cognito authorizer, and every request is checked for an owner here).
//
// A property is a row in the jobs table (input_mode "property", status "generated"),
// so it appears on the Shelf like any model. Its scene (the property document plus
// site, backdrop and weather) is JSON in the private output bucket, and each floor's
// floorplan image sits next to it. Nothing is readable except through this Lambda.
//
// POST { action: "create", title }                          -> { id }
// POST { action: "load", id }                               -> { id, title, scene, imageUrls: { key: url } }
// POST { action: "save", id, scene, title? }                -> { id, updatedAt }
// POST { action: "image", id, levelId, contentType, size }  -> { key, upload } (presigned POST)
// POST { action: "share", id }                              -> { token } (the same one until unshared)
// POST { action: "unshare", id }                            -> { id }
//
// Sharing gives the property a random token; shared.js shows it to anyone with the link
// (/p/<token>) through properties/shares/<token>.json, which names the property and owner.

import crypto from 'node:crypto';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, PutCommand, QueryCommand, UpdateCommand } from '@aws-sdk/lib-dynamodb';
import { GetObjectCommand, PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { createPresignedPost } from '@aws-sdk/s3-presigned-post';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

const region = process.env.AWS_REGION || 'us-east-1';
const dynamodb = DynamoDBDocumentClient.from(new DynamoDBClient({ region }));
const s3 = new S3Client({ region });

const MAX_TITLE_LENGTH = 80;
const MAX_SCENE_BYTES = 2 * 1024 * 1024;
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
const IMAGE_TYPES = { 'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp' };
const UPLOAD_URL_TTL_SECONDS = 15 * 60;
const IMAGE_URL_TTL_SECONDS = 60 * 60;

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

function cleanTitle(value) {
  const title = typeof value === 'string' ? value.replace(/[\u0000-\u001f\u007f]+/g, ' ').trim() : '';
  if (!title) throw new ClientError('Give the property a name.');
  if (title.length > MAX_TITLE_LENGTH) throw new ClientError(`The name must be ${MAX_TITLE_LENGTH} characters or fewer.`);
  return title;
}

class ClientError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.status = status;
  }
}

const sceneKey = (id) => `properties/${id}/scene.json`;
const shareKey = (token) => `properties/shares/${token}.json`;
const validId = (id) => typeof id === 'string' && /^prop-[A-Za-z0-9-]{1,64}$/.test(id);

async function findOwnedProperty(tableName, id, ownerId) {
  if (!validId(id)) throw new ClientError('A valid property id is required.');
  const result = await dynamodb.send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: 'job_id = :id',
    ExpressionAttributeValues: { ':id': id },
    ConsistentRead: true,
    Limit: 2,
  }));
  const item = result.Items?.find((row) => row.owner_id === ownerId && row.input_mode === 'property');
  if (!item) throw new ClientError('Property not found.', 404);
  return item;
}

/** Every floorplan image key the scene refers to, limited to this property's folder. */
function imageKeys(id, scene) {
  const levels = scene?.property?.levels;
  if (!Array.isArray(levels)) return [];
  return levels
    .map((level) => level?.underlay?.imageKey)
    .filter((key) => typeof key === 'string' && key.startsWith(`properties/${id}/plans/`));
}

async function create(tableName, bucket, owner, payload) {
  const title = cleanTitle(payload.title);
  const id = `prop-${crypto.randomUUID()}`;
  const now = new Date().toISOString();
  await dynamodb.send(new PutCommand({
    TableName: tableName,
    Item: {
      job_id: id,
      created_at: now,
      updated_at: now,
      owner_id: owner.userId,
      uploaded_by: owner.uploadedBy,
      title,
      model: 'property',
      input_mode: 'property',
      category: 'building_structure',
      status: 'generated',
      output_bucket: bucket,
      scene_key: sceneKey(id),
    },
    ConditionExpression: 'attribute_not_exists(job_id)',
  }));
  return response(200, { id });
}

async function load(tableName, bucket, owner, payload) {
  const item = await findOwnedProperty(tableName, payload.id, owner.userId);
  let scene = null;
  try {
    const object = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: item.scene_key || sceneKey(item.job_id) }));
    scene = JSON.parse(await object.Body.transformToString());
  } catch (error) {
    // A property that was created but never saved has no scene yet.
    if (error.name !== 'NoSuchKey') throw error;
  }
  const imageUrls = {};
  await Promise.all(imageKeys(item.job_id, scene).map(async (key) => {
    imageUrls[key] = await getSignedUrl(s3, new GetObjectCommand({ Bucket: bucket, Key: key }), {
      expiresIn: IMAGE_URL_TTL_SECONDS,
    });
  }));
  return response(200, { id: item.job_id, title: item.title, scene, imageUrls, shareToken: item.share_token ?? null });
}

async function save(tableName, bucket, owner, payload) {
  const item = await findOwnedProperty(tableName, payload.id, owner.userId);
  const scene = payload.scene;
  if (!scene || typeof scene !== 'object' || !Array.isArray(scene.property?.levels)) {
    throw new ClientError('The scene must contain a property with levels.');
  }
  const body = JSON.stringify(scene);
  if (Buffer.byteLength(body) > MAX_SCENE_BYTES) throw new ClientError('This property is too large to save.');
  const title = payload.title === undefined ? null : cleanTitle(payload.title);

  await s3.send(new PutObjectCommand({
    Bucket: bucket,
    Key: item.scene_key || sceneKey(item.job_id),
    Body: body,
    ContentType: 'application/json',
  }));
  const updatedAt = new Date().toISOString();
  await dynamodb.send(new UpdateCommand({
    TableName: tableName,
    Key: { job_id: item.job_id, created_at: item.created_at },
    UpdateExpression: title ? 'SET updated_at = :now, title = :title' : 'SET updated_at = :now',
    ExpressionAttributeValues: title ? { ':now': updatedAt, ':title': title } : { ':now': updatedAt },
  }));
  return response(200, { id: item.job_id, updatedAt });
}

async function image(tableName, bucket, owner, payload) {
  const item = await findOwnedProperty(tableName, payload.id, owner.userId);
  const extension = IMAGE_TYPES[payload.contentType];
  if (!extension) throw new ClientError('Floorplan images must be JPG, PNG or WEBP.');
  const size = Number(payload.size);
  if (!Number.isInteger(size) || size < 1 || size > MAX_IMAGE_BYTES) {
    throw new ClientError('Floorplan images must be 20 MB or smaller.');
  }
  if (typeof payload.levelId !== 'string' || !/^[A-Za-z0-9_-]{1,64}$/.test(payload.levelId)) {
    throw new ClientError('A valid level id is required.');
  }
  // A new name per upload, so a replaced plan never shows a cached old image.
  const key = `properties/${item.job_id}/plans/${payload.levelId}-${crypto.randomUUID().slice(0, 8)}.${extension}`;
  const upload = await createPresignedPost(s3, {
    Bucket: bucket,
    Key: key,
    Conditions: [
      ['content-length-range', 1, MAX_IMAGE_BYTES],
      ['eq', '$Content-Type', payload.contentType],
    ],
    Fields: { 'Content-Type': payload.contentType },
    Expires: UPLOAD_URL_TTL_SECONDS,
  });
  return response(200, { key, upload });
}

async function share(tableName, bucket, owner, payload) {
  const item = await findOwnedProperty(tableName, payload.id, owner.userId);
  if (item.share_token) return response(200, { token: item.share_token });
  // 128 random bits: links can't be guessed or listed.
  const token = crypto.randomBytes(16).toString('base64url');
  await s3.send(new PutObjectCommand({
    Bucket: bucket,
    Key: shareKey(token),
    Body: JSON.stringify({ propertyId: item.job_id, ownerId: owner.userId, createdAt: new Date().toISOString() }),
    ContentType: 'application/json',
  }));
  await dynamodb.send(new UpdateCommand({
    TableName: tableName,
    Key: { job_id: item.job_id, created_at: item.created_at },
    UpdateExpression: 'SET share_token = :token',
    ExpressionAttributeValues: { ':token': token },
  }));
  return response(200, { token });
}

async function unshare(tableName, bucket, owner, payload) {
  const item = await findOwnedProperty(tableName, payload.id, owner.userId);
  if (item.share_token) {
    // Overwritten rather than deleted (the role can write here, not delete), and the row
    // forgets the token, so shared.js refuses the old link either way.
    await s3.send(new PutObjectCommand({
      Bucket: bucket,
      Key: shareKey(item.share_token),
      Body: JSON.stringify({ revoked: true }),
      ContentType: 'application/json',
    }));
    await dynamodb.send(new UpdateCommand({
      TableName: tableName,
      Key: { job_id: item.job_id, created_at: item.created_at },
      UpdateExpression: 'REMOVE share_token',
    }));
  }
  return response(200, { id: item.job_id });
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
  if (!owner) return response(401, { error: 'Sign in to work on properties.' });

  let payload;
  try {
    payload = getPayload(event);
  } catch {
    return response(400, { error: 'Request body must be valid JSON.' });
  }

  try {
    if (payload.action === 'create') return await create(tableName, bucket, owner, payload);
    if (payload.action === 'load') return await load(tableName, bucket, owner, payload);
    if (payload.action === 'save') return await save(tableName, bucket, owner, payload);
    if (payload.action === 'image') return await image(tableName, bucket, owner, payload);
    if (payload.action === 'share') return await share(tableName, bucket, owner, payload);
    if (payload.action === 'unshare') return await unshare(tableName, bucket, owner, payload);
    return response(400, { error: 'action must be "create", "load", "save", "image", "share" or "unshare".' });
  } catch (error) {
    if (error instanceof ClientError) return response(error.status, { error: error.message });
    console.error('Property request failed:', error);
    return response(500, { error: 'The property request could not be processed.' });
  }
};
