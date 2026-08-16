/**
 * WebRTC tunnel for remote llama.cpp access.
 *
 * Signaling uses the WebTorrent tracker WebSocket protocol (no external deps).
 * The room code is used as the info_hash rendezvous key; the pass code
 * authenticates the client on the data channel after the WebRTC handshake.
 *
 * The client sends an offer, authenticates via pass code, then all same-origin
 * fetch calls are transparently forwarded through the data channel to the host
 * running llama-connect next to its local server.
 */

const STUN_CONFIG: RTCConfiguration = {
	iceServers: [{ urls: 'stun:stun.l.google.com:19302' }, { urls: 'stun:stun1.l.google.com:19302' }]
};
const TRACKER_URLS = ['wss://tracker.openwebtorrent.com', 'wss://tracker.btorrent.xyz'];
const ICE_GATHER_TIMEOUT_MS = 10_000;
const CONNECT_TIMEOUT_MS = 30_000;
const TRACKER_CONNECT_TIMEOUT_MS = 10_000;
// Characters that are unambiguous to read aloud or type
const CODE_CHARS = 'ABCDEFGHJKMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789';

function randomStr(len: number): string {
	const bytes = new Uint8Array(len);

	crypto.getRandomValues(bytes);
	let result = '';

	for (const b of bytes) result += CODE_CHARS[b % CODE_CHARS.length];

	return result;
}

// info_hash must be exactly 20 chars for WebTorrent trackers
function roomToInfoHash(roomCode: string): string {
	return roomCode.padEnd(20, '0').slice(0, 20);
}

function waitForIceComplete(pc: RTCPeerConnection): Promise<void> {
	return new Promise((resolve, reject) => {
		if (pc.iceGatheringState === 'complete') {
			resolve();

			return;
		}

		const timer = setTimeout(
			() => reject(new Error('ICE gathering timed out')),
			ICE_GATHER_TIMEOUT_MS
		);

		pc.addEventListener('icegatheringstatechange', () => {
			if (pc.iceGatheringState === 'complete') {
				clearTimeout(timer);
				resolve();
			}
		});
	});
}

// ---------------------------------------------------------------------------
// Tracker signaling (WebTorrent WS tracker protocol)
// ---------------------------------------------------------------------------

type TrackerMsg = Record<string, unknown>;

class Tracker {
	private ws: WebSocket | null = null;
	private readonly infoHash: string;
	private readonly peerId: string;

	onAnswer?: (offerId: string, answer: RTCSessionDescriptionInit) => void;
	onClose?: () => void;

	constructor(infoHash: string, peerId: string) {
		this.infoHash = infoHash;
		this.peerId = peerId;
	}

	connect(url: string): Promise<void> {
		return new Promise((resolve, reject) => {
			const ws = new WebSocket(url);

			this.ws = ws;
			const timer = setTimeout(
				() => reject(new Error('tracker connect timeout')),
				TRACKER_CONNECT_TIMEOUT_MS
			);

			ws.onopen = () => {
				clearTimeout(timer);
				resolve();
			};
			ws.onerror = () => {
				clearTimeout(timer);
				reject(new Error('tracker WebSocket error'));
			};
			ws.onclose = () => {
				this.onClose?.();
			};
			ws.onmessage = (event) => {
				try {
					const msg = JSON.parse(event.data as string) as TrackerMsg;

					if (msg.answer && msg.offer_id) {
						this.onAnswer?.(msg.offer_id as string, msg.answer as RTCSessionDescriptionInit);
					}
				} catch {
					// ignore malformed tracker messages
				}
			};
		});
	}

	private send(msg: TrackerMsg): void {
		if (this.ws?.readyState === WebSocket.OPEN) {
			this.ws.send(JSON.stringify(msg));
		}
	}

	announce(
		opts: {
			numwant?: number;
			offers?: Array<{ offer_id: string; offer: RTCSessionDescriptionInit }>;
		} = {}
	): void {
		const msg: TrackerMsg = {
			action: 'announce',
			info_hash: this.infoHash,
			numwant: opts.numwant ?? 0,
			peer_id: this.peerId
		};

		if (opts.offers) msg.offers = opts.offers;

		this.send(msg);
	}

	close(): void {
		this.ws?.close();
		this.ws = null;
	}
}

// ---------------------------------------------------------------------------
// Tunnel message types (JSON, sent over RTCDataChannel)
// ---------------------------------------------------------------------------

interface ReqMsg {
	type: 'req';
	id: string;
	method: string;
	path: string;
	headers: Record<string, string>;
	body: string | null; // base64 or null
}
interface ResStartMsg {
	type: 'res_start';
	id: string;
	status: number;
	headers: Record<string, string>;
}
interface ResChunkMsg {
	type: 'res_chunk';
	id: string;
	data: string; // base64
}
interface ResEndMsg {
	type: 'res_end';
	id: string;
}
interface ResErrMsg {
	type: 'res_err';
	id: string;
	message: string;
}
interface CancelMsg {
	type: 'cancel';
	id: string;
}
interface AuthMsg {
	type: 'auth';
	pass: string;
}
interface AuthOkMsg {
	type: 'auth_ok';
}
interface AuthFailMsg {
	type: 'auth_fail';
}

type TunnelMsg =
	| ReqMsg
	| ResStartMsg
	| ResChunkMsg
	| ResEndMsg
	| ResErrMsg
	| CancelMsg
	| AuthMsg
	| AuthOkMsg
	| AuthFailMsg;

function uint8ToBase64(bytes: Uint8Array): string {
	let binary = '';

	for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);

	return btoa(binary);
}

function base64ToUint8(b64: string): Uint8Array {
	return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}
// ---------------------------------------------------------------------------
// ClientTunnel
// ---------------------------------------------------------------------------

export type ClientCallbacks = {
	onConnected?: () => void;
	onDisconnected?: () => void;
};

type PendingReq = {
	onStart: (status: number, headers: Record<string, string>) => void;
	onChunk: (data: string) => void;
	onEnd: () => void;
	onError: (message: string) => void;
};

export class ClientTunnel {
	private readonly passCode: string;
	private readonly infoHash: string;
	private readonly peerId: string;
	private readonly callbacks: ClientCallbacks;

	private pc: RTCPeerConnection | null = null;
	private channel: RTCDataChannel | null = null;
	private tracker: Tracker | null = null;
	private pending = new Map<string, PendingReq>();

	constructor(roomCode: string, passCode: string, callbacks: ClientCallbacks = {}) {
		this.passCode = passCode;
		this.infoHash = roomToInfoHash(roomCode);
		this.peerId = randomStr(20);
		this.callbacks = callbacks;
	}

	get isConnected(): boolean {
		return this.channel?.readyState === 'open';
	}

	async connect(): Promise<void> {
		let lastError: Error = new Error('no trackers available');

		for (const url of TRACKER_URLS) {
			try {
				await this.connectViaTracker(url);

				return;
			} catch (e) {
				lastError = e instanceof Error ? e : new Error(String(e));
				this.cleanupConnection();
			}
		}

		throw lastError;
	}

	private async connectViaTracker(trackerUrl: string): Promise<void> {
		const offerId = randomStr(20);
		const pc = new RTCPeerConnection(STUN_CONFIG);

		this.pc = pc;

		const channel = pc.createDataChannel('tunnel', { ordered: true });

		this.channel = channel;

		const offer = await pc.createOffer();

		await pc.setLocalDescription(offer);
		await waitForIceComplete(pc);

		const tracker = new Tracker(this.infoHash, this.peerId);

		this.tracker = tracker;
		await tracker.connect(trackerUrl);

		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				reject(new Error('connection timed out waiting for host'));
			}, CONNECT_TIMEOUT_MS);

			tracker.onAnswer = async (_offerId, answer) => {
				if (_offerId !== offerId) return;

				try {
					await pc.setRemoteDescription(new RTCSessionDescription(answer));
				} catch (e) {
					clearTimeout(timer);
					reject(e);
				}
			};

			channel.onopen = () => {
				channel.send(JSON.stringify({ pass: this.passCode, type: 'auth' } satisfies AuthMsg));
			};

			channel.onmessage = (event) => {
				try {
					const msg = JSON.parse(event.data as string) as TunnelMsg;

					if (msg.type === 'auth_ok') {
						clearTimeout(timer);
						this.callbacks.onConnected?.();
						resolve();
					} else if (msg.type === 'auth_fail') {
						clearTimeout(timer);
						reject(new Error('authentication failed: invalid passcode'));
					} else {
						this.routeResponseMsg(msg);
					}
				} catch {
					// ignore
				}
			};

			channel.onclose = () => {
				this.callbacks.onDisconnected?.();
				this.rejectAllPending('connection closed');
			};

			channel.onerror = () => {
				clearTimeout(timer);
				reject(new Error('data channel error'));
			};

			tracker.announce({
				numwant: 1,
				offers: [{ offer: pc.localDescription!, offer_id: offerId }]
			});
		});
	}

	private routeResponseMsg(msg: TunnelMsg): void {
		if (
			msg.type !== 'res_start' &&
			msg.type !== 'res_chunk' &&
			msg.type !== 'res_end' &&
			msg.type !== 'res_err'
		)
			return;

		const req = this.pending.get(msg.id);

		if (!req) return;

		if (msg.type === 'res_start') {
			req.onStart(msg.status, msg.headers);
		} else if (msg.type === 'res_chunk') {
			req.onChunk(msg.data);
		} else if (msg.type === 'res_end') {
			req.onEnd();
			this.pending.delete(msg.id);
		} else if (msg.type === 'res_err') {
			req.onError(msg.message);
			this.pending.delete(msg.id);
		}
	}

	private rejectAllPending(reason: string): void {
		for (const req of this.pending.values()) req.onError(reason);
		this.pending.clear();
	}

	private cleanupConnection(): void {
		this.channel?.close();
		this.pc?.close();
		this.tracker?.close();
		this.channel = null;
		this.pc = null;
		this.tracker = null;
	}

	async fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
		if (!this.channel || this.channel.readyState !== 'open') {
			throw new Error('tunnel not connected');
		}

		const request = new Request(input, init);
		const signal = init?.signal ?? (input instanceof Request ? input.signal : undefined);
		const id = randomStr(16);

		if (signal?.aborted) {
			return Promise.reject(new DOMException('Aborted', 'AbortError'));
		}

		// Extract path+query so the host fetches relative to its own origin
		const reqUrl = new URL(request.url);
		const path = reqUrl.pathname + reqUrl.search;
		const headers: Record<string, string> = {};

		request.headers.forEach((v, k) => {
			headers[k] = v;
		});

		let bodyB64: string | null = null;

		const bodyBytes = await request.arrayBuffer();

		if (bodyBytes.byteLength > 0) {
			bodyB64 = uint8ToBase64(new Uint8Array(bodyBytes));
		}

		return new Promise((resolve, reject) => {
			let streamController!: ReadableStreamDefaultController<Uint8Array>;

			const stream = new ReadableStream<Uint8Array>({
				start(ctrl) {
					streamController = ctrl;
				}
			});
			const abortHandler = () => {
				this.pending.delete(id);
				try {
					streamController.error(new DOMException('Aborted', 'AbortError'));
				} catch {
					// stream may already be closed
				}
				reject(new DOMException('Aborted', 'AbortError'));

				// Tell the host to stop the in-flight fetch
				if (this.channel?.readyState === 'open') {
					this.channel.send(JSON.stringify({ id, type: 'cancel' } satisfies CancelMsg));
				}
			};

			signal?.addEventListener('abort', abortHandler, { once: true });

			this.pending.set(id, {
				onChunk: (data) => {
					streamController.enqueue(base64ToUint8(data));
				},
				onEnd: () => {
					signal?.removeEventListener('abort', abortHandler);
					streamController.close();
				},
				onError: (message) => {
					signal?.removeEventListener('abort', abortHandler);
					try {
						streamController.error(new Error(message));
					} catch {
						// stream may already be closed
					}
					reject(new Error(message));
					this.pending.delete(id);
				},
				onStart: (status, resHeaders) => {
					resolve(new Response(stream, { headers: resHeaders, status }));
				}
			});

			this.channel!.send(
				JSON.stringify({
					body: bodyB64,
					headers,
					id,
					method: request.method,
					path,
					type: 'req'
				} satisfies ReqMsg)
			);
		});
	}

	disconnect(): void {
		this.rejectAllPending('disconnected');
		this.cleanupConnection();
	}
}
