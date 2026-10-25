import { S3Client, PutObjectCommand, DeleteObjectCommand } from '@aws-sdk/client-s3';
import { SQSClient, SendMessageCommand } from '@aws-sdk/client-sqs';
import { DynamoDBClient, PutItemCommand, UpdateItemCommand } from '@aws-sdk/client-dynamodb';
import crypto from 'node:crypto';

const region = process.env.AWS_REGION || 'us-east-1';
const sqs = new SQSClient({ region });
const s3 = new S3Client({ region });
const dynamodb = new DynamoDBClient({ region });
const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;
const MAX_TITLE_LENGTH = 80;
const MAX_DESCRIPTION_LENGTH = 500;
const ALLOWED_CATEGORIES = new Set([
  'building_structure',
  'mechanical_device',
  'home_appliance',
  'vehicle',
  'furniture_decor',
  'consumer_product',
  'character_creature',
  'nature_landscape',
  'other',
]);
const CORS_HEADERS = {
  'Access-Control-Allow-Origin': process.env.ALLOWED_ORIGIN || 'http://localhost:5173',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
  'Access-Control-Allow-Methods': 'POST,OPTIONS',
  'Vary': 'Origin',
};

function readRequestBody(event) {
  if (event && typeof event.body === 'object' && event.body !== null) {
    return event.body;
  }

  if (!event || typeof event.body !== 'string' || !event.body.trim()) {
    return {};
  }

  const body = event.isBase64Encoded
    ? Buffer.from(event.body, 'base64').toString('utf8')
    : event.body;
  return JSON.parse(body);
}

function jsonResponse(statusCode, body) {
  return {
    statusCode,
    headers: {
      ...CORS_HEADERS,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  };
}

function getUploader(event) {
  const authorizer = event?.requestContext?.authorizer;
  const claims = authorizer?.jwt?.claims
    || authorizer?.claims
    || authorizer?.lambda
    || {};
  const userId = claims.sub
    || claims.userId
    || claims.principalId
    || claims['cognito:username'];

  if (!userId) {
    return null;
  }

  return {
    userId: String(userId),
    uploadedBy: String(claims.email || claims.preferred_username || claims['cognito:username'] || userId),
  };
}

async function markJobFailed(tableName, jobId, createdAt, error) {
  try {
    await dynamodb.send(new UpdateItemCommand({
      TableName: tableName,
      Key: {
        job_id: { S: jobId },
        created_at: { S: createdAt },
      },
      UpdateExpression: 'SET #status = :status, updated_at = :updatedAt, failure_reason = :reason',
      ExpressionAttributeNames: { '#status': 'status' },
      ExpressionAttributeValues: {
        ':status': { S: 'failed' },
        ':updatedAt': { S: new Date().toISOString() },
        ':reason': { S: String(error?.message || 'Request failed').slice(0, 500) },
      },
    }));
  } catch (statusError) {
    console.error(`Failed to mark job ${jobId} as failed:`, statusError);
  }
}

function decodeImageFile(encoded, contentType) {
  if (
    typeof encoded !== 'string' ||
    !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(encoded)
  ) {
    throw new Error('fileBase64 must contain a valid base64-encoded image.');
  }

  const bytes = Buffer.from(encoded, 'base64');
  if (!bytes.length || bytes.length > MAX_UPLOAD_BYTES) {
    throw new Error(`Image must be between 1 byte and ${MAX_UPLOAD_BYTES} bytes.`);
  }

  const imageTypes = {
    'image/jpeg': {
      extension: 'jpg',
      matches: bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff,
    },
    'image/png': {
      extension: 'png',
      matches: bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])),
    },
    'image/webp': {
      extension: 'webp',
      matches: bytes.toString('ascii', 0, 4) === 'RIFF' && bytes.toString('ascii', 8, 12) === 'WEBP',
    },
  };
  const imageType = imageTypes[contentType];

  if (!imageType || !imageType.matches) {
    throw new Error('Only valid JPEG, PNG, and WebP images are accepted.');
  }

  return { bytes, contentType, extension: imageType.extension };
}

function decodeImage(payload) {
  if (!payload.fileBase64) return null;
  return decodeImageFile(payload.fileBase64, payload.contentType);
}

/** Optional user-entered text: trimmed, control characters removed, length-checked. */
function readOptionalText(value, field, maxLength) {
  if (value === undefined || value === null) return '';
  if (typeof value !== 'string') throw new Error(`${field} must be a string.`);
  const text = value.replace(/[\u0000-\u001f\u007f]+/g, ' ').trim();
  if (text.length > maxLength) throw new Error(`${field} must be ${maxLength} characters or fewer.`);
  return text;
}

export const handler = async (event) => {
  const method = event && (
    event.httpMethod ||
    (event.requestContext && event.requestContext.http && event.requestContext.http.method)
  );

  if (method === 'OPTIONS') {
    return { statusCode: 204, headers: CORS_HEADERS, body: '' };
  }

  if (method && method !== 'POST') {
    return jsonResponse(405, { error: 'Method not allowed.' });
  }

  const queueUrl = process.env.SQS_QUEUE_URL;
  if (!queueUrl) {
    return jsonResponse(500, { error: 'SQS_QUEUE_URL environment variable is not set.' });
  }

  const inputBucket = process.env.INPUT_BUCKET;
  if (!inputBucket) {
    return jsonResponse(500, { error: 'INPUT_BUCKET environment variable is not set.' });
  }

  const tableName = process.env.JOB_STATUS_TABLE;
  if (!tableName) {
    return jsonResponse(500, { error: 'JOB_STATUS_TABLE environment variable is not set.' });
  }

  let payload;
  let image;
  let views;
  let inputMode;
  try {
    payload = readRequestBody(event);
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
      return jsonResponse(400, { error: 'Request body must be a JSON object.' });
    }
    inputMode = payload.inputMode || (payload.fileBase64 ? 'single' : 'prompt');
    if (!['single', 'multiview', 'prompt'].includes(inputMode)) {
      return jsonResponse(400, { error: 'inputMode must be single, multiview, or prompt.' });
    }

    if (inputMode === 'single') {
      image = decodeImage(payload);
    } else if (inputMode === 'multiview') {
      const requiredViews = ['front', 'left', 'back'];
      if (!payload.views || requiredViews.some((view) => !payload.views[view])) {
        return jsonResponse(400, { error: 'Multiview requests require front, left, and back images.' });
      }

      views = {};
      let totalBytes = 0;
      for (const view of requiredViews) {
        const decoded = decodeImageFile(payload.views[view].fileBase64, payload.views[view].contentType);
        totalBytes += decoded.bytes.length;
        if (totalBytes > MAX_UPLOAD_BYTES) {
          return jsonResponse(400, { error: `All reference images combined must be ${MAX_UPLOAD_BYTES} bytes or smaller.` });
        }
        views[view] = decoded;
      }
    }
  } catch (error) {
    return jsonResponse(400, { error: error.message });
  }

  if (inputMode === 'single' && !image) {
    return jsonResponse(400, { error: 'Provide a valid image and contentType.' });
  }
  if (inputMode === 'prompt' && !(typeof payload.prompt === 'string' && payload.prompt.trim())) {
    return jsonResponse(400, { error: 'Provide fileBase64 and contentType, or a prompt.' });
  }
  if (typeof payload.category !== 'string' || !ALLOWED_CATEGORIES.has(payload.category)) {
    return jsonResponse(400, { error: 'Choose a valid model category.' });
  }

  let title;
  let description;
  try {
    title = readOptionalText(payload.title, 'title', MAX_TITLE_LENGTH);
    description = readOptionalText(payload.description, 'description', MAX_DESCRIPTION_LENGTH);
  } catch (error) {
    return jsonResponse(400, { error: error.message });
  }

  if (payload.model !== undefined && payload.model !== null) {
    if (typeof payload.model !== 'string' || !['hunyuan', 'meshy'].includes(payload.model.trim().toLowerCase())) {
      return jsonResponse(400, { error: 'model must be either "hunyuan" or "meshy".' });
    }
  }
  const model = String(payload.model || 'hunyuan').trim().toLowerCase();

  const jobId = payload.jobId || `job-${crypto.randomUUID()}`;
  if (typeof jobId !== 'string' || !/^[A-Za-z0-9_-]{1,64}$/.test(jobId)) {
    return jsonResponse(400, { error: 'jobId may contain only letters, numbers, underscores, and hyphens (up to 64 characters).' });
  }

  const uploader = getUploader(event);
  if (!uploader) {
    return jsonResponse(401, { error: 'Authenticated uploader identity is missing.' });
  }

  const inputKey = image
    ? `uploads/${jobId}/input.${image.extension}`
    : views
      ? `uploads/${jobId}/front.${views.front.extension}`
      : undefined;
  const viewLocations = views
    ? Object.fromEntries(Object.entries(views).map(([view, details]) => [view, {
      bucket: inputBucket,
      key: `uploads/${jobId}/${view}.${details.extension}`,
    }]))
    : undefined;

  const messageBody = {
    jobId,
    model,
    input_mode: inputMode,
    category: payload.category,
    prompt: typeof payload.prompt === 'string' ? payload.prompt : undefined,
    bucket: image ? inputBucket : undefined,
    inputKey: image ? inputKey : undefined,
    contentType: image ? image.contentType : undefined,
    views: viewLocations,
    createdAt: new Date().toISOString(),
    source: 'api-gateway',
  };

  const item = {
    job_id: { S: jobId },
    owner_id: { S: uploader.userId },
    uploaded_by: { S: uploader.uploadedBy },
    model: { S: model },
    input_mode: { S: inputMode },
    category: { S: payload.category },
    status: { S: image || views ? 'uploading' : 'queued' },
    created_at: { S: messageBody.createdAt },
    updated_at: { S: messageBody.createdAt },
  };
  if (image) {
    item.input_bucket = { S: inputBucket };
    item.input_key = { S: inputKey };
    item.content_type = { S: image.contentType };
  }
  if (views) {
    item.input_bucket = { S: inputBucket };
    item.input_key = { S: inputKey };
    item.input_views = {
      M: Object.fromEntries(Object.entries(viewLocations).map(([view, location]) => [view, {
        M: {
          bucket: { S: location.bucket },
          key: { S: location.key },
        },
      }])),
    };
  }
  if (typeof payload.prompt === 'string' && payload.prompt.trim()) {
    item.prompt = { S: payload.prompt.trim() };
  }
  if (title) item.title = { S: title };
  if (description) item.description = { S: description };

  try {
    await dynamodb.send(new PutItemCommand({
      TableName: tableName,
      Item: item,
      ConditionExpression: 'attribute_not_exists(job_id)',
    }));
  } catch (error) {
    console.error('Failed to create job status record:', error);
    if (error.name === 'ConditionalCheckFailedException') {
      return jsonResponse(409, { error: 'A job with this jobId already exists.' });
    }
    return jsonResponse(500, { error: 'Failed to create job status record.' });
  }

  if (image || views) {
    const uploadedKeys = [];
    try {
      if (image) {
        await s3.send(new PutObjectCommand({
          Bucket: inputBucket,
          Key: inputKey,
          Body: image.bytes,
          ContentType: image.contentType,
        }));
        uploadedKeys.push(inputKey);
      }
      if (views) {
        for (const [view, details] of Object.entries(views)) {
          const key = viewLocations[view].key;
          await s3.send(new PutObjectCommand({
            Bucket: inputBucket,
            Key: key,
            Body: details.bytes,
            ContentType: details.contentType,
          }));
          uploadedKeys.push(key);
        }
      }
    } catch (error) {
      console.error('Failed to upload input image(s) to S3:', error);
      await markJobFailed(tableName, jobId, messageBody.createdAt, error);
      for (const key of uploadedKeys) {
        try {
          await s3.send(new DeleteObjectCommand({ Bucket: inputBucket, Key: key }));
        } catch (cleanupError) {
          console.error(`Failed to clean up partial upload ${key}:`, cleanupError);
        }
      }
      return jsonResponse(500, { error: 'Failed to upload image.' });
    }

    try {
      await dynamodb.send(new UpdateItemCommand({
        TableName: tableName,
        Key: {
          job_id: { S: jobId },
          created_at: { S: messageBody.createdAt },
        },
        UpdateExpression: 'SET #status = :status, updated_at = :updatedAt',
        ExpressionAttributeNames: { '#status': 'status' },
        ExpressionAttributeValues: {
          ':status': { S: 'queued' },
          ':updatedAt': { S: new Date().toISOString() },
        },
      }));
    } catch (error) {
      console.error(`Failed to update job ${jobId} after image upload:`, error);
      await markJobFailed(tableName, jobId, messageBody.createdAt, error);
      for (const key of uploadedKeys) {
        try {
          await s3.send(new DeleteObjectCommand({ Bucket: inputBucket, Key: key }));
        } catch (cleanupError) {
          console.error(`Failed to remove unqueued input ${key}:`, cleanupError);
        }
      }
      return jsonResponse(500, { error: 'Failed to update job status.' });
    }
  }

  const params = {
    QueueUrl: queueUrl,
    MessageBody: JSON.stringify(messageBody),
  };

  if (queueUrl.endsWith('.fifo')) {
    params.MessageGroupId = jobId;
    params.MessageDeduplicationId = `${jobId}-${Date.now()}`;
  }

  try {
    const result = await sqs.send(new SendMessageCommand(params));

    return jsonResponse(200, {
      ok: true,
      messageId: result.MessageId,
      jobId,
      model,
      bucket: image ? inputBucket : undefined,
      inputKey,
    });
  } catch (error) {
    console.error('Failed to enqueue SQS job:', error);
    await markJobFailed(tableName, jobId, messageBody.createdAt, error);
    return jsonResponse(500, { ok: false, error: 'Failed to enqueue job.' });
  }
};
