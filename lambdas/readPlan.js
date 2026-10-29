// AI floorplan reading (#54). Signed-in users only (the API route must use the Cognito
// authorizer): it spends money on every call.
//
// POST { image: base64 JPEG / PNG / WEBP, contentType, width, height }
//   -> { reading, coordinates, model }
//
// Sends the plan image to Qwen3.8-27B through an OpenAI-compatible chat API (Groq for
// now; a vLLM server on Modal later, which only changes the env below) and returns what
// it read, cleaned to the shape planReading.js in the app expects:
//   walls [{ a, b, exterior }], openings [{ type, a, b }],
//   rooms [{ name, type, size, polygon }], dimensions [{ text, a, b }]
// with every point in `coordinates` space: "permille" (0-1000 across each side of the
// image, which Qwen's grounding is trained on) or "pixels" (of the image sent).
//
// Env: PLAN_READER_URL (e.g. https://api.groq.com/openai/v1), PLAN_READER_API_KEY,
// PLAN_READER_MODEL (e.g. qwen/qwen3.8-27b), PLAN_READER_COORDINATES (permille | pixels,
// default permille), ALLOWED_ORIGIN. Lambda timeout: 30 s (the API's limit).

const MAX_IMAGE_BYTES = 6 * 1024 * 1024;
const IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
// The API gives up at 30 s; stop the model call before that so the reply says why.
const MODEL_TIMEOUT_MS = 27_000;
const LIMITS = { walls: 400, openings: 300, rooms: 80, dimensions: 120, polygon: 40 };
const ROOM_TYPES = new Set(['drawing', 'lounge', 'bedroom', 'kitchen', 'bath', 'stairs', 'entrance', 'store', 'terrace', 'courtyard', 'other']);

const corsHeaders = {
  'Access-Control-Allow-Origin': process.env.ALLOWED_ORIGIN || 'http://localhost:5173',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
  'Access-Control-Allow-Methods': 'POST,OPTIONS',
  Vary: 'Origin',
};

function response(statusCode, body) {
  return { statusCode, headers: { ...corsHeaders, 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
}

function signedIn(event) {
  const authorizer = event?.requestContext?.authorizer || {};
  const claims = authorizer.jwt?.claims || authorizer.claims || authorizer.lambda || {};
  return Boolean(claims.sub || claims['cognito:username'] || authorizer.principalId);
}

function getPayload(event) {
  if (event?.body && typeof event.body === 'object') return event.body;
  if (typeof event?.body !== 'string' || !event.body.trim()) return {};
  return JSON.parse(event.isBase64Encoded ? Buffer.from(event.body, 'base64').toString('utf8') : event.body);
}

export function instructions(coordinates, width, height) {
  const space = coordinates === 'pixels'
    ? `pixels of this image (${width} x ${height}), x to the right and y down from the top-left corner`
    : 'per mille of the image: x and y from 0 to 1000 across its width and height, from the top-left corner';
  return `You read architectural floorplans of houses (mostly from Pakistan: plans in feet and inches, \
rooms such as Drawing Room, Bed Room, T.V. Lounge, Kitchen, Bath / Toilet, Store, Car Porch, Stairs).

Read the plan in the image and reply with JSON only, in exactly this shape:
{
  "walls": [{ "a": [x, y], "b": [x, y], "exterior": true }],
  "openings": [{ "type": "door", "a": [x, y], "b": [x, y] }],
  "rooms": [{ "name": "Bed Room", "type": "bedroom", "size": "12'-0\\" x 14'-6\\"", "polygon": [[x, y], ...] }],
  "dimensions": [{ "text": "32'-10\\"", "a": [x, y], "b": [x, y] }]
}

Every point is [x, y] in ${space}. The plan may not fill the image: blank white space around it is padding.
- walls: walls are the thick solid bands (often black, grey or hatched). Thin lines are furniture, fixtures, steps, hatching or dimension lines, never walls. Give one straight segment per wall along its centre line, from corner to corner, split where another wall meets it, running straight through any door or window in it. "exterior": true for the outer walls of the house.
- openings: each door (a gap in a wall, usually with a swing arc) and window (thin parallel lines across a wall), "a" and "b" being the two ends of the gap.
- rooms: every room and space, with its label exactly as printed (or a fitting name if none is printed), its type (one of ${[...ROOM_TYPES].join(', ')}), its outline as a polygon along the inside of its walls, and "size": the size printed inside it, copied exactly (e.g. 12'-0" x 14'-6"), or "" when none is printed. Never estimate a size.
- dimensions: only dimension lines that have a printed length, with that text copied exactly and the two ends of what the line measures. Never estimate a length.
Read only one floor (the main or first plan if the sheet has several). Leave out furniture, fixtures, text that isn't a room name or size, hatching and grid lines. If something is unclear, leave it out rather than guess.`;
}

const isNumber = (value) => typeof value === 'number' && Number.isFinite(value);
const point = (value) => (Array.isArray(value) && value.length === 2 && value.every(isNumber) ? value.map((n) => Math.round(n * 10) / 10) : null);
const text = (value, max) => (typeof value === 'string' ? value.replace(/[\u0000-\u001f]+/g, ' ').trim().slice(0, max) : '');

/** The model's JSON, reduced to well-formed entries of the expected shape. */
export function cleanReading(raw) {
  const list = (key) => (Array.isArray(raw?.[key]) ? raw[key].slice(0, LIMITS[key]) : []);
  return {
    walls: list('walls')
      .map((wall) => ({ a: point(wall?.a), b: point(wall?.b), exterior: wall?.exterior === true }))
      .filter((wall) => wall.a && wall.b),
    openings: list('openings')
      .map((opening) => ({ type: opening?.type, a: point(opening?.a), b: point(opening?.b) }))
      .filter((opening) => (opening.type === 'door' || opening.type === 'window') && opening.a && opening.b),
    rooms: list('rooms')
      .map((room) => ({
        name: text(room?.name, 40),
        type: ROOM_TYPES.has(room?.type) ? room.type : 'other',
        size: text(room?.size, 40),
        polygon: (Array.isArray(room?.polygon) ? room.polygon.slice(0, LIMITS.polygon) : []).map(point).filter(Boolean),
      }))
      .filter((room) => room.polygon.length >= 3),
    dimensions: list('dimensions')
      .map((dimension) => ({ text: text(dimension?.text, 24), a: point(dimension?.a), b: point(dimension?.b) }))
      .filter((dimension) => dimension.text && dimension.a && dimension.b),
  };
}

/** The first JSON object in the model's reply (it may wrap it in a code fence or a thought). */
export function parseJson(content) {
  const body = String(content ?? '').replace(/<think>[\s\S]*?<\/think>/g, '');
  const start = body.indexOf('{');
  const end = body.lastIndexOf('}');
  if (start < 0 || end <= start) throw new Error('The model replied without JSON.');
  return JSON.parse(body.slice(start, end + 1));
}

export const handler = async (event) => {
  const method = event?.httpMethod || event?.requestContext?.http?.method;
  if (method === 'OPTIONS') return { statusCode: 204, headers: corsHeaders, body: '' };
  if (method && method !== 'POST') return response(405, { error: 'Method not allowed.' });
  if (!signedIn(event)) return response(401, { error: 'Sign in to read a floorplan.' });

  const { PLAN_READER_URL: url, PLAN_READER_API_KEY: apiKey, PLAN_READER_MODEL: model } = process.env;
  const coordinates = process.env.PLAN_READER_COORDINATES === 'pixels' ? 'pixels' : 'permille';
  if (!url || !apiKey || !model) {
    return response(500, { error: 'PLAN_READER_URL, PLAN_READER_API_KEY and PLAN_READER_MODEL must be set.' });
  }

  let payload;
  try {
    payload = getPayload(event);
  } catch {
    return response(400, { error: 'Request body must be valid JSON.' });
  }
  const { image, contentType } = payload;
  const width = Number(payload.width);
  const height = Number(payload.height);
  if (typeof image !== 'string' || !IMAGE_TYPES.has(contentType)) {
    return response(400, { error: 'Send the plan as a JPG, PNG or WEBP image.' });
  }
  if (Buffer.byteLength(image, 'base64') > MAX_IMAGE_BYTES) return response(400, { error: 'The plan image is too large.' });
  if (!(width > 0 && height > 0)) return response(400, { error: "The image's width and height are required." });

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), MODEL_TIMEOUT_MS);
  const ask = () => fetch(`${url.replace(/\/$/, '')}/chat/completions`, {
      method: 'POST',
      signal: controller.signal,
      headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        temperature: 0,
        max_completion_tokens: 12000,
        response_format: { type: 'json_object' },
        messages: [{
          role: 'user',
          content: [
            { type: 'image_url', image_url: { url: `data:${contentType};base64,${image}` } },
            { type: 'text', text: instructions(coordinates, width, height) },
          ],
        }],
      }),
    });
  try {
    // Now and then the model's JSON comes back malformed even in JSON mode: ask once more.
    let result;
    let raw;
    for (let attempt = 1; attempt <= 2; attempt += 1) {
      const reply = await ask();
      result = await reply.json().catch(() => ({}));
      if (!reply.ok) {
        console.error('Plan reader model error:', reply.status, JSON.stringify(result).slice(0, 500));
        return response(502, { error: "The plan reader couldn't read this plan. Please try again." });
      }
      try {
        raw = parseJson(result.choices?.[0]?.message?.content);
        break;
      } catch (parseError) {
        console.error(`Plan reader reply ${attempt} wasn't JSON:`, parseError.message);
        if (attempt === 2) throw parseError;
      }
    }
    const reading = cleanReading(raw);
    console.log('Plan read:', JSON.stringify({
      model, walls: reading.walls.length, openings: reading.openings.length, rooms: reading.rooms.length,
      dimensions: reading.dimensions.length, usage: result.usage,
    }));
    return response(200, { reading, coordinates, model });
  } catch (error) {
    if (error.name === 'AbortError') return response(504, { error: 'Reading the plan took too long. Please try again.' });
    console.error('Plan reading failed:', error);
    return response(502, { error: "The plan reader couldn't read this plan. Please try again." });
  } finally {
    clearTimeout(timer);
  }
};
