import {
  S3Client,
  GetObjectCommand,
  PutObjectCommand,
  DeleteObjectCommand,
} from '@aws-sdk/client-s3';
import { DynamoDBClient, UpdateItemCommand } from '@aws-sdk/client-dynamodb';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

const region = process.env.AWS_REGION || 'us-east-1';
const s3 = new S3Client({ region });
const dynamodb = new DynamoDBClient({ region });

function isDuplicateMarkerError(error) {
  return error.name === 'PreconditionFailed'
    || error.name === 'ConditionalRequestConflict'
    || error.$metadata?.httpStatusCode === 409
    || error.$metadata?.httpStatusCode === 412;
}

function parseMessageBody(recordBody) {
  if (!recordBody) {
    return {};
  }

  if (typeof recordBody === 'object') {
    return recordBody;
  }

  try {
    const parsed = JSON.parse(recordBody);
    return parsed || {};
  } catch (error) {
    return { rawBody: recordBody };
  }
}

function resolveS3Location(message) {
  if (!message || typeof message !== 'object') {
    return null;
  }

  const payload = message.task || message;
  const bucket = payload.bucket || process.env.INPUT_BUCKET;
  const key = payload.inputKey || payload.s3Key || payload.key;

  if (!bucket || !key) {
    return null;
  }

  return {
    bucket,
    key: String(key).replace(/^\/+/, ''),
    payload,
  };
}

async function streamToBase64(stream) {
  if (stream && typeof stream.transformToString === 'function') {
    return await stream.transformToString('base64');
  }
  const chunks = [];
  for await (const chunk of stream) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
  }
  return Buffer.concat(chunks).toString('base64');
}

async function resolveImageUri({ bucket, key, contentType, s3Client }) {
  if (process.env.MESHY_USE_DATA_URI === 'true') {
    const object = await s3Client.send(new GetObjectCommand({
      Bucket: bucket,
      Key: key,
    }));
    const base64 = await streamToBase64(object.Body);
    const mime = object.ContentType || contentType || 'image/jpeg';
    return `data:${mime};base64,${base64}`;
  }

  const configuredExpiry = Number.parseInt(process.env.MESHY_PRESIGNED_URL_TTL_SECONDS || process.env.PRESIGNED_URL_TTL_SECONDS || '3600', 10);
  const expiresIn = Number.isInteger(configuredExpiry) && configuredExpiry > 0
    ? Math.min(configuredExpiry, 604800)
    : 3600;

  try {
    return await getSignedUrl(s3Client, new GetObjectCommand({
      Bucket: bucket,
      Key: key,
    }), { expiresIn });
  } catch (signError) {
    console.warn(`Failed to generate presigned URL for s3://${bucket}/${key}, falling back to Base64 data URI:`, signError);
    const object = await s3Client.send(new GetObjectCommand({
      Bucket: bucket,
      Key: key,
    }));
    const base64 = await streamToBase64(object.Body);
    const mime = object.ContentType || contentType || 'image/jpeg';
    return `data:${mime};base64,${base64}`;
  }
}

function sanitizePayloadForLog(payload) {
  if (!payload || typeof payload !== 'object') return payload;
  const copy = { ...payload };
  if (typeof copy.image_url === 'string' && copy.image_url.startsWith('data:')) {
    copy.image_url = `${copy.image_url.slice(0, 30)}... [${copy.image_url.length} chars]`;
  }
  if (Array.isArray(copy.image_urls)) {
    copy.image_urls = copy.image_urls.map((u) => (
      typeof u === 'string' && u.startsWith('data:')
        ? `${u.slice(0, 30)}... [${u.length} chars]`
        : u
    ));
  }
  return copy;
}

async function updateJobStatus(jobId, createdAt, status, failureReason, extraAttributes = {}) {
  const tableName = process.env.JOB_STATUS_TABLE;
  if (!tableName) throw new Error('JOB_STATUS_TABLE environment variable is not set.');
  if (typeof createdAt !== 'string' || !createdAt) {
    throw new Error(`Missing createdAt sort key for job ${jobId}.`);
  }

  const expressionValues = {
    ':status': { S: status },
    ':updatedAt': { S: new Date().toISOString() },
  };
  const expressionNames = { '#status': 'status' };
  const setActions = ['#status = :status', 'updated_at = :updatedAt'];
  const removeActions = [];

  if (failureReason) {
    setActions.push('failure_reason = :reason');
    expressionValues[':reason'] = { S: String(failureReason).slice(0, 500) };
  } else if (status === 'processing') {
    removeActions.push('failure_reason');
  }

  for (const [key, val] of Object.entries(extraAttributes)) {
    if (val !== undefined && val !== null) {
      const safeKey = key.replace(/[^a-zA-Z0-9]/g, '_');
      const attrName = `#attr_${safeKey}`;
      const attrVal = `:val_${safeKey}`;
      setActions.push(`${attrName} = ${attrVal}`);
      expressionNames[attrName] = key;
      expressionValues[attrVal] = { S: String(val) };
    }
  }

  let updateExpression = `SET ${setActions.join(', ')}`;
  if (removeActions.length > 0) {
    updateExpression += ` REMOVE ${removeActions.join(', ')}`;
  }

  await dynamodb.send(new UpdateItemCommand({
    TableName: tableName,
    Key: {
      job_id: { S: jobId },
      created_at: { S: createdAt },
    },
    UpdateExpression: updateExpression,
    ExpressionAttributeNames: expressionNames,
    ExpressionAttributeValues: expressionValues,
    ConditionExpression: 'attribute_exists(job_id)',
  }));
}

export const handler = async (event) => {
  const records = Array.isArray(event && event.Records) ? event.Records : [];
  const batchItemFailures = [];
  const tableName = process.env.JOB_STATUS_TABLE;
  if (!tableName) throw new Error('JOB_STATUS_TABLE environment variable is not set.');

  for (const record of records) {
    let markerKey;
    let outputBucket;
    let requestStarted = false;
    let jobId;
    let createdAt;
    let model;

    try {
      const message = parseMessageBody(record.body);
      const payload = message.task || message;
      const location = resolveS3Location(message);
      const inputMode = payload.input_mode
        || (payload.views ? 'multiview' : location ? 'single' : 'prompt');
      const views = payload.views;
      jobId = payload.jobId;
      createdAt = payload.createdAt;
      model = String(payload.model || 'hunyuan').trim().toLowerCase();

      if (typeof jobId !== 'string' || !/^[A-Za-z0-9_-]{1,64}$/.test(jobId)) {
        throw new Error(`Invalid or missing jobId in messageId: ${record.messageId}`);
      }

      if (model !== 'hunyuan' && model !== 'meshy') {
        throw new Error(`Unsupported model "${payload.model}". Supported models are "hunyuan" and "meshy".`);
      }

      if (inputMode === 'multiview') {
        const hasValidHunyuanViews = views && ['front', 'left', 'back'].every((view) => views[view]?.bucket && views[view]?.key);
        const hasValidMeshyUrls = Array.isArray(payload.image_urls) && payload.image_urls.length > 0;
        const hasValidMeshyViews = views && (
          Array.isArray(views) ? views.length > 0 : ['front', 'left', 'back'].some((view) => views[view]?.bucket && views[view]?.key)
        );

        if (model === 'hunyuan' && !hasValidHunyuanViews) {
          throw new Error(`Multiview message must include front, left, and back S3 references: ${record.messageId}`);
        }
        if (model === 'meshy' && !hasValidMeshyUrls && !hasValidMeshyViews) {
          throw new Error(`Multiview message must include valid image references: ${record.messageId}`);
        }
      } else if (inputMode === 'single') {
        const hasSingleLocation = Boolean(location);
        const hasDirectUrl = Boolean(payload.image_url || payload.imageUrl);
        if (!hasSingleLocation && !hasDirectUrl) {
          throw new Error(`Missing S3 bucket or inputKey in messageId: ${record.messageId}`);
        }
      } else if (inputMode === 'prompt') {
        if (!(typeof payload.prompt === 'string' && payload.prompt.trim())) {
          throw new Error(`Missing prompt in messageId: ${record.messageId}`);
        }
      }

      outputBucket = process.env.OUTPUT_BUCKET;
      if (!outputBucket) {
        throw new Error('OUTPUT_BUCKET environment variable is not set.');
      }

      if (model === 'hunyuan') {
        if (!process.env.MODAL_ENDPOINT_URL) {
          throw new Error('MODAL_ENDPOINT_URL environment variable is not set.');
        }
      } else if (model === 'meshy') {
        const meshyApiKey = process.env.MESHY_API_KEY || process.env.MESHY_API_TOKEN;
        if (!meshyApiKey) {
          throw new Error('MESHY_API_KEY environment variable is not set.');
        }
      }

      markerKey = `_idempotency/${jobId}.json`;
      try {
        await s3.send(new PutObjectCommand({
          Bucket: outputBucket,
          Key: markerKey,
          Body: JSON.stringify({
            jobId,
            messageId: record.messageId,
            acceptedAt: new Date().toISOString(),
          }),
          ContentType: 'application/json',
          IfNoneMatch: '*',
        }));
      } catch (error) {
        if (isDuplicateMarkerError(error)) {
          console.log(`Skipping duplicate SQS delivery for job ${jobId}.`);
          continue;
        }
        throw error;
      }

      await updateJobStatus(jobId, createdAt, 'processing', undefined, { model });

      if (model === 'hunyuan') {
        let fileSize;
        let contentType = payload.contentType || 'application/octet-stream';
        const viewMetadata = {};
        if (inputMode === 'multiview') {
          fileSize = 0;
          for (const view of ['front', 'left', 'back']) {
            const object = await s3.send(new GetObjectCommand({
              Bucket: views[view].bucket,
              Key: views[view].key,
            }));
            fileSize += object.ContentLength || 0;
            viewMetadata[view] = { bucket: views[view].bucket, key: views[view].key };
          }
        } else if (location) {
          const object = await s3.send(new GetObjectCommand({
            Bucket: location.bucket,
            Key: location.key,
          }));
          fileSize = object.ContentLength || 0;
          contentType = object.ContentType || contentType;
        }

        const modalPayload = {
          prompt: payload.prompt,
          messageId: record.messageId,
          jobId,
          input_mode: inputMode,
          bucket: location?.bucket,
          key: location?.key,
          views: inputMode === 'multiview' ? viewMetadata : undefined,
          output_bucket: outputBucket,
        };

        console.log(JSON.stringify({
          ...modalPayload,
          contentType,
          fileSize,
        }));

        requestStarted = true;
        const response = await fetch(process.env.MODAL_ENDPOINT_URL, {
          method: 'POST',
          headers: {
            'Modal-Key': process.env.MODAL_KEY,
            'Modal-Secret': process.env.MODAL_SECRET,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(modalPayload),
        });
        const responseBody = await response.text();
        if (!response.ok) {
          await s3.send(new DeleteObjectCommand({
            Bucket: outputBucket,
            Key: markerKey,
          }));
          markerKey = undefined;
          throw new Error(`Modal endpoint returned ${response.status}: ${responseBody}`);
        }

        console.log(`Modal API response for job ${jobId}: ${responseBody}`);
      } else if (model === 'meshy') {
        const meshyApiKey = process.env.MESHY_API_TOKEN;
        const meshyBaseUrl = (process.env.MESHY_BASE_URL || 'https://api.meshy.ai').replace(/\/+$/, '');
        let endpointUrl;
        let meshyPayload;

        if (inputMode === 'multiview') {
          endpointUrl = `${meshyBaseUrl}/openapi/v1/multi-image-to-3d`;

          let imageUrls = [];
          if (Array.isArray(payload.image_urls) && payload.image_urls.length > 0) {
            imageUrls = payload.image_urls.slice(0, 3);
          } else if (Array.isArray(payload.imageUrls) && payload.imageUrls.length > 0) {
            imageUrls = payload.imageUrls.slice(0, 3);
          } else if (Array.isArray(views)) {
            const selected = views.slice(0, 3);
            imageUrls = await Promise.all(selected.map((v) => {
              if (typeof v === 'string') return v;
              return resolveImageUri({
                bucket: v.bucket || location?.bucket || process.env.INPUT_BUCKET,
                key: v.key || v.inputKey,
                contentType: v.contentType,
                s3Client: s3,
              });
            }));
          } else if (views && typeof views === 'object') {
            const preferredViews = ['front', 'left', 'back'].filter((k) => views[k]?.bucket && views[k]?.key);
            const keysToUse = preferredViews.length > 0 ? preferredViews : Object.keys(views).filter((k) => views[k]?.bucket && views[k]?.key);
            const selectedKeys = keysToUse.slice(0, 3);
            imageUrls = await Promise.all(selectedKeys.map((k) => {
              const v = views[k];
              if (typeof v === 'string') return v;
              return resolveImageUri({
                bucket: v.bucket,
                key: v.key,
                contentType: v.contentType,
                s3Client: s3,
              });
            }));
          }

          if (!imageUrls || imageUrls.length === 0) {
            throw new Error(`Multiview message must include valid image references: ${record.messageId}`);
          }

          meshyPayload = {
            image_urls: imageUrls,
            ...(payload.ai_model ? { ai_model: payload.ai_model } : {}),
          };
        } else if (inputMode === 'single') {
          endpointUrl = `${meshyBaseUrl}/openapi/v1/image-to-3d`;

          let imageUrl = payload.image_url || payload.imageUrl;
          if (!imageUrl && location) {
            imageUrl = await resolveImageUri({
              bucket: location.bucket,
              key: location.key,
              contentType: payload.contentType,
              s3Client: s3,
            });
          }

          if (!imageUrl) {
            throw new Error(`Missing image for single image-to-3d in messageId: ${record.messageId}`);
          }

          meshyPayload = {
            image_url: imageUrl,
            ...(payload.ai_model ? { ai_model: payload.ai_model } : {}),
          };
        } else if (inputMode === 'prompt') {
          endpointUrl = `${meshyBaseUrl}/openapi/v2/text-to-3d`;

          meshyPayload = {
            mode: payload.mode || 'preview',
            prompt: payload.prompt.trim(),
            ...(payload.ai_model ? { ai_model: payload.ai_model } : {}),
          };
        }

        console.log(`Submitting Meshy task for job ${jobId}:`, JSON.stringify(sanitizePayloadForLog(meshyPayload)));

        requestStarted = true;
        const response = await fetch(endpointUrl, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${meshyApiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(meshyPayload),
        });

        const responseBody = await response.text();
        if (!response.ok) {
          await s3.send(new DeleteObjectCommand({
            Bucket: outputBucket,
            Key: markerKey,
          }));
          markerKey = undefined;
          throw new Error(`Meshy endpoint returned ${response.status}: ${responseBody}`);
        }

        console.log(`Meshy API response for job ${jobId}: ${responseBody}`);

        let resultData;
        try {
          resultData = JSON.parse(responseBody);
        } catch {
          resultData = null;
        }

        if (resultData?.result) {
          try {
            await updateJobStatus(jobId, createdAt, 'processing', undefined, {
              meshy_task_id: resultData.result,
              model,
            });
          } catch (statusError) {
            console.warn(`Failed to update meshy_task_id for job ${jobId}:`, statusError);
          }
        }
      }
    } catch (error) {
      if (markerKey && !requestStarted) {
        try {
          await s3.send(new DeleteObjectCommand({
            Bucket: outputBucket,
            Key: markerKey,
          }));
        } catch (cleanupError) {
          console.error(`Failed to release idempotency marker ${markerKey}:`, cleanupError);
        }
      }
      if (jobId && /^[A-Za-z0-9_-]{1,64}$/.test(jobId)) {
        try {
          await updateJobStatus(jobId, createdAt, 'failed', error.message);
        } catch (statusError) {
          console.error(`Failed to mark job ${jobId} as failed:`, statusError);
        }
      }
      batchItemFailures.push({ itemIdentifier: record.messageId });
      console.error(`Failed to process SQS message ${record.messageId}:`, error);
    }
  }

  return {
    batchItemFailures,
  };
};
