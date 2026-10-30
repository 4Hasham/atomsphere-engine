import { ConditionalCheckFailedException, DynamoDBClient, QueryCommand, UpdateItemCommand } from '@aws-sdk/client-dynamodb';
import { GetObjectCommand, S3Client } from '@aws-sdk/client-s3';

const dynamodb = new DynamoDBClient({ region: process.env.AWS_REGION || 'us-east-1' });
const s3 = new S3Client({ region: process.env.AWS_REGION || 'us-east-1' });

// Said when a failure note can't be read (or gives no reason).
const GENERIC_FAILURE = "The 3D model couldn't be made this time. Please try again.";

function decodeS3Key(key) {
  return decodeURIComponent(key.replace(/\+/g, ' '));
}

/** The job's status row key: { job_id, created_at }. */
async function jobKey(tableName, jobId) {
  const query = await dynamodb.send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: 'job_id = :jobId',
    ExpressionAttributeValues: { ':jobId': { S: jobId } },
    ProjectionExpression: 'job_id, created_at',
    ConsistentRead: true,
    Limit: 2,
  }));
  if (!query.Items?.length) {
    throw new Error(`No job status row found for job ${jobId}.`);
  }
  if (query.Items.length > 1) {
    throw new Error(`Multiple status rows found for job_id ${jobId}; cannot determine created_at sort key.`);
  }
  const createdAt = query.Items[0].created_at?.S;
  if (!createdAt) {
    throw new Error(`Job status row ${jobId} has no created_at sort key.`);
  }
  return { job_id: { S: jobId }, created_at: { S: createdAt } };
}

/** The reason in a failure note the generator left (assets/<job>.failed.json). */
async function failureReason(bucket, key) {
  try {
    const object = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
    const note = JSON.parse(await object.Body.transformToString());
    if (note.error) console.log(`Generator error for ${key}: ${note.error}`);
    return String(note.reason || GENERIC_FAILURE).slice(0, 500);
  } catch (error) {
    console.error(`Couldn't read failure note ${key}:`, error);
    return GENERIC_FAILURE;
  }
}

/**
 * S3 ObjectCreated in the output bucket: a finished model (assets/<job>.glb) marks its
 * job generated; a failure note the generator leaves instead (assets/<job>.failed.json)
 * marks it failed, with the reason, unless the model was made after all.
 */
export const handler = async (event) => {
  const tableName = process.env.JOB_STATUS_TABLE;
  const outputBucket = process.env.OUTPUT_BUCKET;
  if (!tableName || !outputBucket) {
    throw new Error('JOB_STATUS_TABLE and OUTPUT_BUCKET environment variables are required.');
  }

  const records = Array.isArray(event?.Records) ? event.Records : [];
  for (const record of records) {
    if (!record.eventName?.startsWith('ObjectCreated:') || !record.s3) {
      continue;
    }

    const bucket = record.s3.bucket.name;
    const key = decodeS3Key(record.s3.object.key);
    if (bucket !== outputBucket) {
      continue;
    }

    const match = /^assets\/([A-Za-z0-9_-]{1,64})\.(glb|failed\.json)$/.exec(key);
    if (!match) {
      continue;
    }
    const [, jobId, kind] = match;
    const Key = await jobKey(tableName, jobId);

    if (kind === 'glb') {
      await dynamodb.send(new UpdateItemCommand({
        TableName: tableName,
        Key,
        UpdateExpression: 'SET #status = :status, output_bucket = :bucket, output_key = :key, updated_at = :updatedAt',
        ExpressionAttributeNames: { '#status': 'status' },
        ExpressionAttributeValues: {
          ':status': { S: 'generated' },
          ':bucket': { S: bucket },
          ':key': { S: key },
          ':updatedAt': { S: new Date().toISOString() },
        },
        ConditionExpression: 'attribute_exists(job_id)',
      }));
      continue;
    }

    try {
      await dynamodb.send(new UpdateItemCommand({
        TableName: tableName,
        Key,
        UpdateExpression: 'SET #status = :status, failure_reason = :reason, updated_at = :updatedAt',
        ExpressionAttributeNames: { '#status': 'status' },
        ExpressionAttributeValues: {
          ':status': { S: 'failed' },
          ':reason': { S: await failureReason(bucket, key) },
          ':updatedAt': { S: new Date().toISOString() },
          ':generated': { S: 'generated' },
        },
        ConditionExpression: 'attribute_exists(job_id) AND #status <> :generated',
      }));
    } catch (error) {
      if (!(error instanceof ConditionalCheckFailedException)) throw error;
      console.log(`Job ${jobId} was generated after all; ignoring its failure note.`);
    }
  }

  return { statusCode: 200, body: JSON.stringify({ ok: true }) };
};
