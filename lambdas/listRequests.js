import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, QueryCommand } from '@aws-sdk/lib-dynamodb';

const region = process.env.AWS_REGION || 'us-east-1';
const dynamodb = DynamoDBDocumentClient.from(new DynamoDBClient({ region }));
const corsHeaders = {
  'Access-Control-Allow-Origin': process.env.ALLOWED_ORIGIN || 'http://localhost:5173',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
  'Access-Control-Allow-Methods': 'GET,OPTIONS',
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

export const handler = async (event) => {
  const method = event?.httpMethod || event?.requestContext?.http?.method;
  if (method === 'OPTIONS') return { statusCode: 204, headers: corsHeaders, body: '' };
  if (method && method !== 'GET') return response(405, { error: 'Method not allowed.' });

  const tableName = process.env.JOB_STATUS_TABLE;
  const indexName = process.env.JOB_OWNER_INDEX || 'owner_id-created_at-index';
  const ownerId = getOwnerId(event);
  if (!tableName) return response(500, { error: 'JOB_STATUS_TABLE environment variable is not set.' });
  if (!ownerId) return response(401, { error: 'Authenticated user identity is missing.' });

  try {
    const jobs = [];
    let lastEvaluatedKey;

    do {
      const result = await dynamodb.send(new QueryCommand({
        TableName: tableName,
        IndexName: indexName,
        KeyConditionExpression: 'owner_id = :ownerId',
        ExpressionAttributeValues: { ':ownerId': ownerId },
        // Placeholders for every newer field, so none can clash with a DynamoDB reserved word.
        ProjectionExpression: 'job_id, #status, created_at, updated_at, prompt, input_key, output_bucket, output_key, '
          + '#title, #description, #category, #input_mode, #model, #size_m, failure_reason',
        ExpressionAttributeNames: {
          '#status': 'status',
          '#title': 'title',
          '#description': 'description',
          '#category': 'category',
          '#input_mode': 'input_mode',
          '#model': 'model',
          '#size_m': 'size_m',
        },
        ScanIndexForward: false,
        ExclusiveStartKey: lastEvaluatedKey,
      }));

      jobs.push(...(result.Items || []));
      lastEvaluatedKey = result.LastEvaluatedKey;
    } while (lastEvaluatedKey);

    return response(200, { jobs });
  } catch (error) {
    console.error('Failed to list user jobs:', error);
    return response(500, { error: 'Failed to list requests.' });
  }
};