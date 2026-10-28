// A shared property, for anyone with its link (/p/<token>). Public: the API route must
// NOT use the Cognito authorizer. Read-only, and it returns only what a viewer needs.
//
// GET /shared/{token} -> { title, scene, modelUrls: { jobId: url }, hiddenObjects }
//
// The token is made by properties.js ("share") and points at properties/shares/<token>.json
// ({ propertyId, ownerId }); unsharing overwrites that and removes the token from the
// property's row, and both are checked here.
//
// Objects placed in the property are included only if they belong to its owner, are
// finished, and were made with Meshy (Polished) or uploaded. Hunyuan3D (Standard) outputs
// may not be shown in the EU, UK or South Korea, and a link can be opened anywhere, so
// they're left out; `hiddenObjects` says how many. Floorplan images are never included.
//
// Env: JOB_STATUS_TABLE, OUTPUT_BUCKET, ALLOWED_ORIGIN. IAM: dynamodb:Query on the table;
// s3:GetObject on properties/* and on the model keys (assets/*, user-models/*).

import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, QueryCommand } from '@aws-sdk/lib-dynamodb';
import { GetObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

const region = process.env.AWS_REGION || 'us-east-1';
const dynamodb = DynamoDBDocumentClient.from(new DynamoDBClient({ region }));
const s3 = new S3Client({ region });

const MODEL_URL_TTL_SECONDS = 60 * 60;
const MAX_OBJECTS = 200;
// Models that may be shown anywhere: Meshy (Polished) and people's own uploads.
const SHOWABLE_MODELS = new Set(['meshy', 'upload']);

const corsHeaders = {
  'Access-Control-Allow-Origin': process.env.ALLOWED_ORIGIN || 'http://localhost:5173',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Allow-Methods': 'GET,OPTIONS',
  Vary: 'Origin',
};

function response(statusCode, body) {
  return {
    statusCode,
    // Signed links inside: never cached.
    headers: { ...corsHeaders, 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
    body: JSON.stringify(body),
  };
}

const NOT_SHARED = { error: "This property isn't shared, or the link has been turned off." };

async function readJson(bucket, key) {
  try {
    const object = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
    return JSON.parse(await object.Body.transformToString());
  } catch (error) {
    if (error.name === 'NoSuchKey') return null;
    throw error;
  }
}

async function findJob(tableName, jobId) {
  const result = await dynamodb.send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: 'job_id = :id',
    ExpressionAttributeValues: { ':id': jobId },
    Limit: 2,
  }));
  return result.Items?.[0] ?? null;
}

/** The property document without anything only the owner needs (floorplan images). */
function viewerProperty(property) {
  return {
    ...property,
    levels: (property?.levels ?? []).map((level) => ({ ...level, underlay: undefined })),
  };
}

export const handler = async (event) => {
  const method = event?.httpMethod || event?.requestContext?.http?.method;
  if (method === 'OPTIONS') return { statusCode: 204, headers: corsHeaders, body: '' };
  if (method && method !== 'GET') return response(405, { error: 'Method not allowed.' });

  const tableName = process.env.JOB_STATUS_TABLE;
  const bucket = process.env.OUTPUT_BUCKET;
  if (!tableName || !bucket) {
    return response(500, { error: 'JOB_STATUS_TABLE and OUTPUT_BUCKET environment variables are required.' });
  }

  const token = event?.pathParameters?.token;
  if (typeof token !== 'string' || !/^[A-Za-z0-9_-]{16,64}$/.test(token)) return response(404, NOT_SHARED);

  try {
    const share = await readJson(bucket, `properties/shares/${token}.json`);
    if (!share?.propertyId || share.revoked) return response(404, NOT_SHARED);

    const property = await findJob(tableName, share.propertyId);
    if (!property || property.input_mode !== 'property' || property.owner_id !== share.ownerId
      || property.share_token !== token) {
      return response(404, NOT_SHARED);
    }

    const scene = await readJson(bucket, property.scene_key || `properties/${property.job_id}/scene.json`);
    if (!scene?.property) return response(404, NOT_SHARED);

    const placed = (Array.isArray(scene.objects) ? scene.objects : [])
      .filter((object) => object?.placement && typeof object.source?.jobId === 'string'
        && /^[A-Za-z0-9_-]{1,64}$/.test(object.source.jobId))
      .slice(0, MAX_OBJECTS);

    const modelUrls = {};
    await Promise.all([...new Set(placed.map((object) => object.source.jobId))].map(async (jobId) => {
      const job = await findJob(tableName, jobId);
      if (!job || job.owner_id !== share.ownerId || job.status !== 'generated'
        || !job.output_bucket || !job.output_key || !SHOWABLE_MODELS.has(job.model)) {
        return;
      }
      modelUrls[jobId] = await getSignedUrl(s3, new GetObjectCommand({
        Bucket: job.output_bucket,
        Key: job.output_key,
        ResponseContentType: 'model/gltf-binary',
      }), { expiresIn: MODEL_URL_TTL_SECONDS });
    }));
    const shown = placed.filter((object) => modelUrls[object.source.jobId]);

    return response(200, {
      title: property.title,
      scene: {
        version: scene.version,
        site: scene.site ?? null,
        backdrop: scene.backdrop ?? 'none',
        weather: scene.weather ?? null,
        objects: shown,
        property: viewerProperty(scene.property),
      },
      modelUrls,
      hiddenObjects: placed.length - shown.length,
    });
  } catch (error) {
    console.error(`Shared property ${token} failed:`, error);
    return response(500, { error: "Couldn't open this property. Please try again." });
  }
};
