'use strict';

// Replay a verified final mask on *new* RGB pixels. This helper imports only
// the installed healer and TIFF codec; it never runs the defect detector and
// never edits the installed application or an existing signed review.
const fs = require('node:fs/promises');
const fsSync = require('node:fs');
const path = require('node:path');
const { setImmediate: yieldNow } = require('node:timers/promises');
const controller = new AbortController();
function cancel() {
  controller.abort();
  // The worker and helper share their cancellation process group. If the
  // Python owner is terminated too, its finally block cannot remove scratch.
  // Reclaim only this helper's strictly named, private temporary directory.
  const requestPath = path.resolve(process.argv[2] || '');
  const scratch = path.dirname(requestPath);
  if (path.basename(requestPath) === 'request.json' && path.basename(scratch).startsWith('.repair-replay-')) {
    try { fsSync.rmSync(scratch, {recursive: true, force: true}); } catch {}
  }
}
process.on('SIGTERM', cancel);
process.on('SIGINT', cancel);

function emit(value) { process.stdout.write(JSON.stringify(value) + '\n'); }
function tick() {
  if (controller.signal.aborted) throw new Error('除尘重放已取消');
  return yieldNow().then(() => {
    if (controller.signal.aborted) throw new Error('除尘重放已取消');
  });
}

async function main() {
  const request = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
  const root = path.resolve(request.backend);
  const { decodeTiffForWork, decodeTiff, encodeTiff } = require(path.join(root, 'src/tiff.cjs'));
  const source = await decodeTiffForWork(await fs.readFile(request.input));
  const coverage = decodeTiff(await fs.readFile(request.mask));
  const W = source.width, H = source.height, C = source.channels;
  if (source.depth !== 16 || C !== 3 || source.colorChannels !== 3 || source.alpha ||
      !source.icc?.length || coverage.width !== W || coverage.height !== H ||
      coverage.channels !== 1 || coverage.depth !== 8) throw new Error('除尘重放输入尺寸、RGB16、ICC或蒙版无效');
  const mask = Uint8Array.from(coverage.data, value => value ? 1 : 0);
  let pixels = 0;
  for (const value of mask) pixels += value;
  emit({type: 'progress', fraction: 0.12, message: '按已确认蒙版重放修补', mask_pixels: pixels});
  let healed;
  if (W * H > 4_194_304) {
    const { healLarge } = require(path.join(root, 'vendor/film-dust/engine/bigheal.js'));
    const io = {
      read: async (x0, y0, x1, y1) => {
        await tick();
        const width = x1 - x0, out = new Uint16Array(width * (y1 - y0) * C);
        for (let y = y0; y < y1; y++) {
          out.set(source.data.subarray((y * W + x0) * C, (y * W + x1) * C), (y - y0) * width * C);
        }
        return out;
      },
      write: async (x0, y0, x1, y1, values) => {
        const width = x1 - x0;
        for (let y = y0; y < y1; y++) {
          source.data.set(values.subarray((y - y0) * width * C, (y - y0 + 1) * width * C), (y * W + x0) * C);
        }
        await tick();
      },
    };
    healed = await healLarge(mask, null, W, H, 1, W, H, C, io, {
      tick,
      progress: fraction => emit({type: 'progress', fraction: 0.12 + 0.72 * fraction, message: '按原像素分块重放修补'}),
    });
  } else {
    const { heal } = require(path.join(root, 'vendor/film-dust/engine/heal.js'));
    healed = await heal(source.data, W, H, C, mask, null,
      fraction => emit({type: 'progress', fraction: 0.12 + 0.72 * fraction, message: '重放已确认修补'}), tick);
  }
  await tick();
  await fs.writeFile(request.output, encodeTiff(source), {flag: 'wx', mode: 0o600});
  emit({type: 'result', mask_pixels: pixels, healed, detector_calls: 0,
    algorithm: W * H > 4_194_304 ? 'PURE Neg healLarge k=1' : 'PURE Neg heal'});
}

main().catch(error => {
  emit({type: 'error', message: error.message || String(error)});
  process.exitCode = 1;
});
