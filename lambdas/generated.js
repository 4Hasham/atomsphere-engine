import { DynamoDBClient, QueryCommand, UpdateItemCommand } from '@aws-sdk/client-dynamodb';

const dynamodb = new DynamoDBClient({ region: process.env.AWS_REGION || 'us-east-1' });

function decodeS3Key(key) {
  return decodeURIComponent(key.replace(/\+/g, ' '));
}

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

    const match = /^assets\/([A-Za-z0-9_-]{1,64})\.glb$/.exec(key);
    if (!match) {
      continue;
    }
    const jobId = match[1];

    const query = await dynamodb.send(new QueryCommand({
      TableName: tableName,
      KeyConditionExpression: 'job_id = :jobId',
      ExpressionAttributeValues: { ':jobId': { S: jobId } },
      ProjectionExpression: 'job_id, created_at',
      ConsistentRead: true,
      Limit: 2,
    }));
    if (!query.Items?.length) {
      throw new Error(`No job status row found for generated model ${jobId}.`);
    }
    if (query.Items.length > 1) {
      throw new Error(`Multiple status rows found for job_id ${jobId}; cannot determine created_at sort key.`);
    }
    const createdAt = query.Items[0].created_at?.S;
    if (!createdAt) {
      throw new Error(`Job status row ${jobId} has no created_at sort key.`);
    }

    await dynamodb.send(new UpdateItemCommand({
      TableName: tableName,
      Key: {
        job_id: { S: jobId },
        created_at: { S: createdAt },
      },
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
  }

  return { statusCode: 200, body: JSON.stringify({ ok: true }) };
};
