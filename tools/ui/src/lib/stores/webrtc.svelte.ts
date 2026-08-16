import { browser } from '$app/environment';
import { ClientTunnel } from '$lib/utils/webrtc-tunnel';

// Stores the active session for auto-reconnect on reload.
const SESSION_KEY = 'llama_webrtc_session';

type SessionData = { roomCode: string; passCode: string };
type ConnectionStatus = 'idle' | 'connecting' | 'connected' | 'error';

class WebRTCStore {
	mode = $state<'off' | 'client'>('off');
	status = $state<ConnectionStatus>('idle');
	errorMessage = $state('');

	private clientTunnel: ClientTunnel | null = null;
	// Requests that arrive while mode='client' but tunnel not yet open are held here.
	private connectionWaiters: Array<{ resolve: () => void; reject: (e: Error) => void }> = [];
	// The original window.fetch saved before the interceptor is installed.
	private originalFetch: typeof window.fetch | null = null;

	constructor() {
		if (browser) {
			this.restoreSession();
		}
	}

	get isConnected(): boolean {
		return this.status === 'connected';
	}

	async joinAsClient(shareCode: string): Promise<void> {
		if (shareCode.length < 40) throw new Error('Invalid code: must be 40 characters');

		const roomCode = shareCode.slice(0, 8);
		const passCode = shareCode.slice(8);

		await this.activateClient(roomCode, passCode);
	}

	private async activateClient(roomCode: string, passCode: string): Promise<void> {
		this.mode = 'client';
		this.status = 'connecting';
		this.errorMessage = '';
		// Install the fetch interceptor synchronously (before any await) so that
		// requests fired by layout effects on the same tick are already captured.
		this.installInterceptor();

		const tunnel = new ClientTunnel(roomCode, passCode, {
			onConnected: () => {
				this.status = 'connected';
			},
			onDisconnected: () => {
				this.status = 'error';
				this.errorMessage = 'Disconnected from host';
			}
		});

		try {
			await tunnel.connect();
			this.clientTunnel = tunnel;
			this.writeSession({ passCode, roomCode });
			// Release any requests that were queued while connecting.
			const waiters = this.connectionWaiters.splice(0);

			for (const w of waiters) w.resolve();
		} catch (e) {
			this.clientTunnel = null;
			this.mode = 'off';
			this.status = 'error';
			this.errorMessage = e instanceof Error ? e.message : String(e);
			this.uninstallInterceptor();
			// Reject queued requests.
			const waiters = this.connectionWaiters.splice(0);
			const err = e instanceof Error ? e : new Error(String(e));

			for (const w of waiters) w.reject(err);

			throw e;
		}
	}

	leaveAsClient(): void {
		this.uninstallInterceptor();
		this.clientTunnel?.disconnect();
		this.clientTunnel = null;
		this.mode = 'off';
		this.status = 'idle';
		this.clearSession();
	}

	// -------------------------------------------------------------------------
	// Fetch interceptor (installed synchronously when client mode activates)
	// -------------------------------------------------------------------------

	private installInterceptor(): void {
		if (this.originalFetch) return; // already installed

		this.originalFetch = window.fetch.bind(window);
		window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
			try {
				const url =
					input instanceof Request ? input.url : input instanceof URL ? input.href : String(input);
				const parsed = new URL(url, window.location.href);

				if (parsed.origin === window.location.origin) {
					return this.tunnelFetch(input, init);
				}
			} catch {
				// not a parseable URL — fall through
			}

			return this.originalFetch!(input, init);
		};
	}

	private uninstallInterceptor(): void {
		if (!this.originalFetch) return;

		window.fetch = this.originalFetch;
		this.originalFetch = null;
	}

	// -------------------------------------------------------------------------
	// Fetch proxy
	// -------------------------------------------------------------------------

	tunnelFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
		// If the tunnel is open, forward immediately.
		if (this.clientTunnel?.isConnected) {
			return this.clientTunnel.fetch(input, init);
		}

		// If we are still connecting, queue the request until the tunnel opens.
		if (this.mode === 'client' && this.status === 'connecting') {
			return new Promise<void>((resolve, reject) => {
				this.connectionWaiters.push({ reject, resolve });
			}).then(() => this.clientTunnel!.fetch(input, init));
		}

		throw new Error('tunnel not connected');
	}

	// -------------------------------------------------------------------------
	// Persistence helpers
	// -------------------------------------------------------------------------

	private restoreSession(): void {
		try {
			const raw = localStorage.getItem(SESSION_KEY);

			if (!raw) return;

			const session = JSON.parse(raw) as SessionData;

			void this.activateClient(session.roomCode, session.passCode);
		} catch {
			// ignore corrupt storage
		}
	}

	private writeSession(data: SessionData): void {
		localStorage.setItem(SESSION_KEY, JSON.stringify(data));
	}

	private clearSession(): void {
		localStorage.removeItem(SESSION_KEY);
	}
}

export const webrtcStore = new WebRTCStore();
