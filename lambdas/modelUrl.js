import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, QueryCommand } from '@aws-sdk/lib-dynamodb';
import { GetObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

const region = process.env.AWS_REGION || 'us-east-1';
const dynamodb = DynamoDBDocumentClient.from(new DynamoDBClient({ region }));
const s3 = new S3Client({ region });
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

function getOwnerId(event) {
  const authorizer = event?.requestContext?.authorizer || {};
  const claims = authorizer.jwt?.claims || authorizer.claims || authorizer.lambda || {};
  const ownerId = claims.sub
    || claims.userId
    || claims.principalId
    || claims['cognito:username']
    || authorizer.principalId;
  return ownerId ? String(ownerId) : null;
}

function getPayload(event) {
  if (event?.body && typeof event.body === 'object') return event.body;
  if (typeof event?.body !== 'string' || !event.body.trim()) return {};
  return JSON.parse(event.isBase64Encoded
    ? Buffer.from(event.body, 'base64').toString('utf8')
    : event.body);
}

export const handler = async (event) => {
  const method = event?.httpMethod || event?.requestContext?.http?.method;
  if (method === 'OPTIONS') return { statusCode: 204, headers: corsHeaders, body: '' };
  if (method && method !== 'POST') return response(405, { error: 'Method not allowed.' });

  const tableName = process.env.JOB_STATUS_TABLE;
  const ownerId = getOwnerId(event);
  if (!tableName) return response(500, { error: 'JOB_STATUS_TABLE environment variable is not set.' });
  if (!ownerId) return response(401, { error: 'Authenticated user identity is missing.' });

  let payload;
  try {
    payload = getPayload(event);
  } catch {
    return response(400, { error: 'Request body must be valid JSON.' });
  }

  const jobId = payload.job_id || payload.jobId;
  if (typeof jobId !== 'string' || !/^[A-Za-z0-9_-]{1,64}$/.test(jobId)) {
    return response(400, { error: 'A valid job_id is required.' });
  }

  try {
    const result = await dynamodb.send(new QueryCommand({
      TableName: tableName,
      KeyConditionExpression: 'job_id = :jobId',
      ExpressionAttributeValues: { ':jobId': jobId },
      ConsistentRead: true,
      ProjectionExpression: 'job_id, created_at, owner_id, #status, output_bucket, output_key',
      ExpressionAttributeNames: { '#status': 'status' },
      Limit: 2,
    }));
    const job = result.Items?.find((item) => item.owner_id === ownerId);

    if (!job || job.owner_id !== ownerId) {
      return response(404, { error: 'Request not found.' });
    }
    if (job.status !== 'generated' || !job.output_bucket || !job.output_key) {
      return response(409, { error: 'The model is not ready yet.' });
    }

    const configuredExpiry = Number.parseInt(process.env.PRESIGNED_URL_TTL_SECONDS || '300', 10);
    const expiresIn = Number.isInteger(configuredExpiry) && configuredExpiry > 0
      ? Math.min(configuredExpiry, 604800)
      : 300;
    const url = await getSignedUrl(s3, new GetObjectCommand({
      Bucket: job.output_bucket,
      Key: job.output_key,
      ResponseContentType: 'model/gltf-binary',
      ResponseContentDisposition: 'inline',
    }), { expiresIn });

    return response(200, {
      job_id: jobId,
      url,
      expires_in: expiresIn,
    });
  } catch (error) {
    console.error(`Failed to create model URL for job ${jobId}:`, error);
    return response(500, { error: 'Failed to create model URL.' });
  }
};