import {
  DynamoDBClient,
  QueryCommand,
  ScanCommand,
  GetItemCommand,
  UpdateItemCommand,
} from '@aws-sdk/client-dynamodb';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import crypto from 'node:crypto';
import { Readable } from 'node:stream';

// ---------- Config (hardcoded) ----------
const MESHY_BASE_URL = 'https://api.meshy.ai';
const MESHY_TASK_ID_INDEX = 'meshy_task_id-index'; // GSI name convention
const ALLOWED_MODEL_HOSTS = ['meshy.ai'];
const API_FETCH_TIMEOUT_MS = 15 * 1000;
const MODEL_FETCH_TIMEOUT_MS = 120 * 1000;
const MAX_MODEL_BYTES = 300 * 1024 * 1024;
const STALE_LOCK_MS = 3 * 60 * 1000;

// ---------- AWS clients (region auto-detected from Lambda env) ----------
const dynamodb = new DynamoDBClient({});
const s3 = new S3Client({
  requestStreamBufferSize: 64 * 1024, // 64 KiB buffer
});

// ---------- Utilities ----------
function jsonResponse(statusCode, body) {
  return {
    statusCode,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
}

function safeEqual(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string') return false;
  const ba = Buffer.from(a);
  const bb = Buffer.from(b);
  if (ba.length !== bb.length) return false;
  return crypto.timingSafeEqual(ba, bb);
}

async function fetchWithTimeout(url, options = {}, timeoutMs = API_FETCH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function parseRequestBody(event) {
  if (!event) return {};
  if (typeof event === 'object' && (event.id || event.result) && !event.body) return event;
  if (event.body && typeof event.body === 'object') return event.body;
  if (typeof event.body === 'string' && event.body.trim()) {
    const raw = event.isBase64Encoded
      ? Buffer.from(event.body, 'base64').toString('utf8')
      : event.body;
    try {
      return JSON.parse(raw);
    } catch {
      return { rawBody: raw };
    }
  }
  return {};
}

function getRawBodyString(event) {
  if (typeof event?.body !== 'string') return null;
  return event.isBase64Encoded
    ? Buffer.from(event.body, 'base64').toString('utf8')
    : event.body;
}

/**
 * API Gateway (REST + HTTP), ALB, and Lambda Function URLs all set requestContext
 * on the event. Direct Lambda.Invoke does not. This is how we distinguish the
 * webhook (public) path from the internal async-worker path without a shared token.
 */
function isHttpTriggered(event) {
  return !!(event?.requestContext || event?.httpMethod);
}

// ---------- Webhook auth ----------
function verifyWebhookSecret(event, body) {
  const expected = process.env.MESHY_WEBHOOK_SECRET;
  if (!expected) {
    console.warn('MESHY_WEBHOOK_SECRET not set; accepting request without verification.');
    return true;
  }
  const headers = Object.fromEntries(
    Object.entries(event?.headers || {}).map(([k, v]) => [k.toLowerCase(), v])
  );

  const headerSecret =
    headers['x-meshy-api-webhook-secret-key'];
  if (headerSecret && safeEqual(headerSecret, expected)) return true;

  const auth = headers['authorization'];
  if (auth) {
    const token = auth.replace(/^Bearer\s+/i, '').trim();
    if (safeEqual(token, expected)) return true;
  }

  const q = event?.queryStringParameters || {};
  const qSecret = q.secret || q.token || q.webhook_secret;
  if (qSecret && safeEqual(qSecret, expected)) return true;

  const sig =
    headers['x-signature'] ||
    headers['x-meshy-signature'] ||
    headers['x-hub-signature-256'] ||
    headers['signature'];
  if (sig) {
    const raw = getRawBodyString(event);
    if (raw) {
      const hex = crypto.createHmac('sha256', expected).update(raw).digest('hex');
      const b64 = crypto.createHmac('sha256', expected).update(raw).digest('base64');
      const clean = sig.replace(/^sha256=/i, '').trim();
      if (safeEqual(clean, hex) || safeEqual(clean, b64)) return true;
    } else {
      console.warn('Signature header present but raw body unavailable; skipping HMAC check.');
    }
  }

  if (body && (body.secret === expected || body.webhook_secret === expected)) return true;
  return false;
}

// ---------- Status mapping ----------
function canonicalStatus(raw) {
  const u = String(raw || '').trim().toUpperCase();
  if (u === 'SUCCEEDED') return 'SUCCEEDED';
  if (u === 'FAILED') return 'FAILED';
  if (u === 'CANCELED' || u === 'CANCELLED') return 'CANCELED';
  if (u === 'EXPIRED') return 'EXPIRED';
  if (u === 'IN_PROGRESS' || u === 'PROCESSING' || u === 'RUNNING') return 'IN_PROGRESS';
  if (u === 'PENDING' || u === 'QUEUED') return 'PENDING';
  return null;
}

function statusRank(c) {
  if (c === 'PENDING') return 0;
  if (c === 'IN_PROGRESS') return 1;
  if (c === 'SUCCEEDED' || c === 'FAILED' || c === 'CANCELED' || c === 'EXPIRED') return 2;
  return -1;
}

function canonicalToDbStatus(c) {
  switch (c) {
    case 'PENDING': return 'pending';
    case 'IN_PROGRESS': return 'processing';
    case 'SUCCEEDED': return 'generated';
    default: return 'failed';
  }
}

// ---------- Job lookup ----------
async function getJobByKey(tableName, jobId, createdAt) {
  const res = await dynamodb.send(new GetItemCommand({
    TableName: tableName,
    Key: { job_id: { S: jobId }, created_at: { S: createdAt } },
    ConsistentRead: true,
  }));
  return res.Item || null;
}

async function findJobRecord(tableName, jobId, taskId) {
  const projection =
    'job_id, created_at, input_mode, meshy_task_id, #status, status_rank, processed_delivery_keys';
  const names = { '#status': 'status' };

  if (jobId) {
    const q = await dynamodb.send(new QueryCommand({
      TableName: tableName,
      KeyConditionExpression: 'job_id = :jobId',
      ExpressionAttributeValues: { ':jobId': { S: jobId } },
      ExpressionAttributeNames: names,
      ProjectionExpression: projection,
      ConsistentRead: true,
      Limit: 2,
    }));
    if (q.Items?.length) return q.Items[0];
  }
  if (!taskId) return null;

  const byTaskQuery = await dynamodb.send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: 'job_id = :jobId',
    ExpressionAttributeValues: { ':jobId': { S: taskId } },
    ExpressionAttributeNames: names,
    ProjectionExpression: projection,
    ConsistentRead: true,
    Limit: 2,
  }));
  if (byTaskQuery.Items?.length) return byTaskQuery.Items[0];

  try {
    const g = await dynamodb.send(new QueryCommand({
      TableName: tableName,
      IndexName: MESHY_TASK_ID_INDEX,
      KeyConditionExpression: 'meshy_task_id = :taskId',
      ExpressionAttributeValues: { ':taskId': { S: taskId } },
      ExpressionAttributeNames: names,
      ProjectionExpression: projection,
      Limit: 2,
    }));
    if (g.Items?.length) return g.Items[0];
  } catch (err) {
    console.warn(`GSI ${MESHY_TASK_ID_INDEX} unavailable (${err.message}); falling back to scan.`);
  }

  // Last-resort paginated scan (bounded: max 5 pages). Prefer adding the GSI.
  let lastKey;
  for (let page = 0; page < 5; page++) {
    const scan = await dynamodb.send(new ScanCommand({
      TableName: tableName,
      FilterExpression: 'meshy_task_id = :taskId',
      ExpressionAttributeValues: { ':taskId': { S: taskId } },
      ExpressionAttributeNames: names,
      ProjectionExpression: projection,
      ExclusiveStartKey: lastKey,
    }));
    if (scan.Items?.length) return scan.Items[0];
    if (!scan.LastEvaluatedKey) break;
    lastKey = scan.LastEvaluatedKey;
  }
  return null;
}

// ---------- Meshy API ----------
function resolveTaskGetEndpoint(taskId, type, inputMode) {
  const t = String(type || '').toLowerCase();
  const m = String(inputMode || '').toLowerCase();
  if (t.includes('multi-image') || m === 'multiview') {
    return `${MESHY_BASE_URL}/openapi/v1/multi-image-to-3d/${taskId}`;
  }
  if (t.includes('image-to-3d') || m === 'single') {
    return `${MESHY_BASE_URL}/openapi/v1/image-to-3d/${taskId}`;
  }
  if (t.includes('text-to-3d') || m === 'prompt') {
    return `${MESHY_BASE_URL}/openapi/v2/text-to-3d/${taskId}`;
  }
  return null;
}

async function fetchMeshyTaskDetails(taskId, type, inputMode) {
  const apiKey = process.env.MESHY_API_KEY;
  if (!apiKey) return null;
  const headers = { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` };

  const specific = resolveTaskGetEndpoint(taskId, type, inputMode);
  if (specific) {
    try {
      const r = await fetchWithTimeout(specific, { method: 'GET', headers });
      if (r.ok) return await r.json();
    } catch (err) {
      console.warn(`GET ${specific} failed: ${err.message}`);
    }
  }
  const probes = [
    `${MESHY_BASE_URL}/openapi/v2/text-to-3d/${taskId}`,
    `${MESHY_BASE_URL}/openapi/v1/image-to-3d/${taskId}`,
    `${MESHY_BASE_URL}/openapi/v1/multi-image-to-3d/${taskId}`,
  ];
  for (const url of probes) {
    if (url === specific) continue;
    try {
      const r = await fetchWithTimeout(url, { method: 'GET', headers });
      if (r.ok) return await r.json();
    } catch (err) {
      console.warn(`Probe ${url} failed: ${err.message}`);
    }
  }
  return null;
}

// ---------- GLB download + upload (SSRF-safe, streamed) ----------
function assertTrustedModelUrl(urlStr) {
  let u;
  try { u = new URL(urlStr); } catch { throw new Error('Invalid model URL'); }
  if (u.protocol !== 'https:') throw new Error('Model URL must be https');
  const host = u.hostname.toLowerCase();
  const ok = ALLOWED_MODEL_HOSTS.some(a => host === a || host.endsWith(`.${a}`));
  if (!ok) throw new Error(`Untrusted model host: ${host}`);
}

async function downloadAndUploadGlb(glbUrl, bucket, key) {
  assertTrustedModelUrl(glbUrl);
  const res = await fetchWithTimeout(glbUrl, {}, MODEL_FETCH_TIMEOUT_MS);
  if (!res.ok) throw new Error(`GLB download failed: ${res.status} ${res.statusText}`);

  const clHeader = res.headers.get('content-length');
  const contentLength = clHeader ? parseInt(clHeader, 10) : null;
  if (contentLength && contentLength > MAX_MODEL_BYTES) {
    throw new Error(`GLB too large: ${contentLength} bytes`);
  }

  let body, length;
  if (res.body && contentLength) {
    body = Readable.fromWeb(res.body);
    length = contentLength;
  } else {
    const ab = await res.arrayBuffer();
    if (ab.byteLength > MAX_MODEL_BYTES) throw new Error(`GLB too large: ${ab.byteLength} bytes`);
    body = Buffer.from(ab);
    length = ab.byteLength;
  }

  await s3.send(new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    Body: body,
    ContentLength: length,
    ContentType: 'model/gltf-binary',
    CacheControl: 'public, max-age=31536000, immutable',
  }));
}

// ---------- Async self-invoke ----------
let lambdaClientPromise = null;
function getLambdaClient() {
  if (!lambdaClientPromise) {
    lambdaClientPromise = import('@aws-sdk/client-lambda')
      .then(({ LambdaClient }) => new LambdaClient({}))
      .catch(err => {
        console.warn('@aws-sdk/client-lambda unavailable:', err.message);
        return null;
      });
  }
  return lambdaClientPromise;
}

async function enqueueProcessing(payload) {
  const fnName = process.env.AWS_LAMBDA_FUNCTION_NAME;
  if (!fnName) return false;
  const client = await getLambdaClient();
  if (!client) return false;
  try {
    const { InvokeCommand } = await import('@aws-sdk/client-lambda');
    await client.send(new InvokeCommand({
      FunctionName: fnName,
      InvocationType: 'Event',
      Payload: Buffer.from(JSON.stringify(payload)),
    }));
    return true;
  } catch (err) {
    console.error('Async invoke failed:', err);
    return false;
  }
}

async function releaseLock(tableName, jobId, createdAt) {
  try {
    await dynamodb.send(new UpdateItemCommand({
      TableName: tableName,
      Key: { job_id: { S: jobId }, created_at: { S: createdAt } },
      UpdateExpression: 'REMOVE processing_lock',
    }));
  } catch (err) {
    console.warn('releaseLock failed:', err.message);
  }
}

// ---------- SUCCEEDED (fast path — webhook side) ----------
async function handleSucceededDelivery(tableName, jobRecord, ctx) {
  const { jobId, createdAt } = ctx;
  if ((jobRecord.status?.S || '').toLowerCase() === 'generated') {
    return jsonResponse(200, { ok: true, jobId, status: 'generated', deduplicated: true });
  }
  try {
    await dynamodb.send(new UpdateItemCommand({
      TableName: tableName,
      Key: { job_id: { S: jobId }, created_at: { S: createdAt } },
      UpdateExpression: 'SET processing_lock = :now, updated_at = :now',
      ConditionExpression:
        'attribute_exists(job_id) AND #status <> :generated ' +
        'AND (attribute_not_exists(processing_lock) OR processing_lock < :stale)',
      ExpressionAttributeNames: { '#status': 'status' },
      ExpressionAttributeValues: {
        ':now': { S: new Date().toISOString() },
        ':generated': { S: 'generated' },
        ':stale': { S: new Date(Date.now() - STALE_LOCK_MS).toISOString() },
      },
    }));
  } catch (err) {
    if (err.name === 'ConditionalCheckFailedException') {
      return jsonResponse(200, { ok: true, jobId, accepted: true, note: 'already_processing' });
    }
    throw err;
  }

  const payload = {
    __task: 'processSucceeded',
    taskId: ctx.taskId,
    jobId,
    createdAt,
    taskType: ctx.taskType,
    glbUrl: ctx.body?.model_urls?.glb || null,
    thumbnailUrl: ctx.body?.thumbnail_url || null,
  };

  if (await enqueueProcessing(payload)) {
    return jsonResponse(202, { ok: true, accepted: true, deferred: true, jobId });
  }
  console.warn('Async enqueue failed; falling back to synchronous processing.');
  try {
    const result = await processSucceededTask(payload);
    return jsonResponse(200, { ok: true, jobId, ...result });
  } catch (err) {
    await releaseLock(tableName, jobId, createdAt);
    throw err;
  }
}

// ---------- SUCCEEDED (slow worker — async invoke) ----------
async function processSucceededTask(payload) {
  const { taskId, jobId, createdAt, taskType } = payload;
  const tableName = process.env.JOB_STATUS_TABLE;
  const outputBucket = process.env.OUTPUT_BUCKET;
  if (!tableName || !outputBucket) throw new Error('Missing table/bucket config');

  try {
    const jobRecord = await getJobByKey(tableName, jobId, createdAt);
    if (!jobRecord) throw new Error(`Job ${jobId} not found during async processing`);
    if ((jobRecord.status?.S || '').toLowerCase() === 'generated') {
      await releaseLock(tableName, jobId, createdAt);
      return { skipped: true, reason: 'already_generated' };
    }

    let glbUrl = payload.glbUrl;
    let thumbnailUrl = payload.thumbnailUrl || null;
    if (!glbUrl) {
      const taskData = await fetchMeshyTaskDetails(taskId, taskType, jobRecord.input_mode?.S);
      if (!taskData) throw new Error('Could not fetch task details from Meshy');
      const status = (taskData.status || '').toUpperCase();
      if (status !== 'SUCCEEDED') {
        await releaseLock(tableName, jobId, createdAt);
        return { skipped: true, reason: 'task_not_succeeded', status };
      }
      glbUrl = taskData.model_urls?.glb;
      thumbnailUrl = taskData.thumbnail_url || thumbnailUrl;
    }
    if (!glbUrl) throw new Error('SUCCEEDED task has no GLB URL');

    const outputKey = `assets/${jobId}.glb`;
    await downloadAndUploadGlb(glbUrl, outputBucket, outputKey);

    const deliveryKey = `${taskId}#SUCCEEDED`;
    const setParts = [
      '#status = :generated', '#statusRank = :rank',
      'output_bucket = :bucket', 'output_key = :key',
      'updated_at = :now', 'last_seen_at = :now',
    ];
    const values = {
      ':generated': { S: 'generated' },
      ':rank': { N: '2' },
      ':bucket': { S: outputBucket },
      ':key': { S: outputKey },
      ':now': { S: new Date().toISOString() },
      ':dk': { SS: [deliveryKey] },
    };
    if (thumbnailUrl) {
      setParts.push('thumbnail_url = :thumb');
      values[':thumb'] = { S: thumbnailUrl };
    }
    const updateExpr =
      `SET ${setParts.join(', ')} REMOVE processing_lock, failure_reason ADD processed_delivery_keys :dk`;

    try {
      await dynamodb.send(new UpdateItemCommand({
        TableName: tableName,
        Key: { job_id: { S: jobId }, created_at: { S: createdAt } },
        UpdateExpression: updateExpr,
        ConditionExpression: '#status <> :generated',
        ExpressionAttributeNames: { '#status': 'status', '#statusRank': 'status_rank' },
        ExpressionAttributeValues: values,
      }));
    } catch (err) {
      if (err.name === 'ConditionalCheckFailedException') {
        await releaseLock(tableName, jobId, createdAt);
        return { skipped: true, reason: 'concurrent_generated' };
      }
      throw err;
    }
    return { status: 'generated', outputKey };
  } catch (err) {
    await releaseLock(tableName, jobId, createdAt);
    throw err;
  }
}

// ---------- Non-SUCCEEDED status update ----------
async function applyStatusUpdate(tableName, ctx) {
  const { canonical, deliveryKey, jobId, createdAt, body } = ctx;
  const newRank = statusRank(canonical);
  const dbStatus = canonicalToDbStatus(canonical);
  const isFailure = ['FAILED', 'CANCELED', 'EXPIRED'].includes(canonical);

  const setParts = [
    '#status = :newStatus', '#statusRank = :newRank',
    'updated_at = :now', 'last_seen_at = :now',
  ];
  const values = {
    ':newStatus': { S: dbStatus },
    ':newRank': { N: String(newRank) },
    ':now': { S: new Date().toISOString() },
    ':dk': { SS: [deliveryKey] },
    ':generated': { S: 'generated' },
    ':failed': { S: 'failed' },
  };
  if (isFailure) {
    setParts.push('failure_reason = :reason');
    const reason =
      body?.task_error?.message || body?.message || `Meshy task ${canonical.toLowerCase()}`;
    values[':reason'] = { S: String(reason).slice(0, 500) };
  }

  const condition =
    'attribute_exists(job_id) ' +
    'AND (attribute_not_exists(processed_delivery_keys) ' +
    '     OR NOT contains(processed_delivery_keys, :dk)) ' +
    'AND #status <> :generated ' +
    'AND #status <> :failed ' +
    'AND (attribute_not_exists(#statusRank) OR #statusRank < :newRank)';

  try {
    await dynamodb.send(new UpdateItemCommand({
      TableName: tableName,
      Key: { job_id: { S: jobId }, created_at: { S: createdAt } },
      UpdateExpression: `SET ${setParts.join(', ')} ADD processed_delivery_keys :dk`,
      ConditionExpression: condition,
      ExpressionAttributeNames: { '#status': 'status', '#statusRank': 'status_rank' },
      ExpressionAttributeValues: values,
    }));
    return jsonResponse(200, { ok: true, jobId, status: dbStatus });
  } catch (err) {
    if (err.name === 'ConditionalCheckFailedException') {
      return jsonResponse(200, { ok: true, jobId, deduplicated: true, note: 'already_applied_or_out_of_order' });
    }
    throw err;
  }
}

// ---------- Webhook entry ----------
async function handleWebhook(event) {
  const tableName = process.env.JOB_STATUS_TABLE;
  const outputBucket = process.env.OUTPUT_BUCKET;
  if (!tableName || !outputBucket) {
    console.error('Missing JOB_STATUS_TABLE or OUTPUT_BUCKET');
    return jsonResponse(500, { error: 'Server misconfigured' });
  }

  let body;
  try { body = parseRequestBody(event); }
  catch (err) {
    console.error('Body parse failed:', err);
    return jsonResponse(400, { error: 'Request body must be valid JSON.' });
  }

  if (!verifyWebhookSecret(event, body)) {
    console.warn('Webhook secret verification failed.');
    return jsonResponse(401, { error: 'Unauthorized' });
  }

  const taskId = body.id || body.task_id || event?.queryStringParameters?.id;
  let jobId =
    body.job_id || body.jobId ||
    event?.queryStringParameters?.job_id || event?.queryStringParameters?.jobId;
  const taskType = body.type || body.task_type;

  if (!taskId && !jobId) {
    return jsonResponse(400, { error: 'Missing task id or jobId.' });
  }

  const canonical = canonicalStatus(body.status);
  if (!canonical) {
    // Unknown status → ack (200) so Meshy doesn't retry unnecessarily
    console.warn(`Unknown status '${body.status}' for ${taskId || jobId}; acking.`);
    return jsonResponse(200, { ok: true, ignored: true, reason: 'unknown_status' });
  }

  console.log(`Meshy webhook: taskId=${taskId || '-'} jobId=${jobId || '-'} status=${canonical}`);

  const jobRecord = await findJobRecord(tableName, jobId, taskId);
  if (!jobRecord) {
    // Transient race — return 404 so Meshy retries until the row exists.
    console.warn(`No job record for taskId=${taskId} jobId=${jobId}; returning 404 to trigger retry.`);
    return jsonResponse(404, { error: `Job not found for taskId ${taskId || jobId}` });
  }

  jobId = jobRecord.job_id?.S || jobId;
  const createdAt = jobRecord.created_at?.S;
  if (!createdAt) throw new Error(`Job ${jobId} missing created_at sort key.`);

  const deliveryKey = `${taskId || jobId}#${canonical}`;
  if ((jobRecord.processed_delivery_keys?.SS || []).includes(deliveryKey)) {
    return jsonResponse(200, { ok: true, jobId, deduplicated: true });
  }

  try {
    if (canonical === 'SUCCEEDED') {
      return await handleSucceededDelivery(tableName, jobRecord, {
        taskId, jobId, createdAt, taskType, body,
      });
    }
    return await applyStatusUpdate(tableName, {
      canonical, deliveryKey, jobId, createdAt, body,
    });
  } catch (err) {
    console.error(`Error processing webhook for job ${jobId}:`, err);
    return jsonResponse(500, { error: err.message });
  }
}

// ---------- Router ----------
export const handler = async (event) => {
  // Direct Lambda.Invoke (no requestContext) = internal worker path.
  // Publicly reachable HTTP triggers always set requestContext, so they
  // can never reach this branch.
  if (!isHttpTriggered(event)) {
    if (event?.__task === 'processSucceeded') {
      try {
        return await processSucceededTask(event);
      } catch (err) {
        console.error('Async processing failed:', err);
        throw err; // Lambda async retry → DLQ
      }
    }
    return { statusCode: 400, body: JSON.stringify({ error: 'Unknown internal task' }) };
  }
  return await handleWebhook(event);
};