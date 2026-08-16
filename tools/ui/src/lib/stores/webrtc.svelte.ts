import { browser } from '$app/environment';
import { CODE_LENGTHS } from '$lib/constants';
import { IS_WEB_ONLY } from '$lib/constants/web-only.constants';
import { RemoteAccessMode, TunnelStatus } from '$lib/enums';
import { ClientTunnel } from '$lib/utils/webrtc-tunnel';

// Stores the active session for auto-reconnect on reload.
const SESSION_KEY = 'llama_webrtc_session';

type SessionData = { roomCode: string; passCode: string };

class WebRTCStore {
	mode = $state<RemoteAccessMode>(RemoteAccessMode.OFF);
	status = $state<TunnelStatus>(TunnelStatus.IDLE);
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
		return this.status === TunnelStatus.CONNECTED;
	}

	/**
	 * A web-only build has no server of its own, so it stays unusable until a
	 * code is stored. Restoring a saved session already sets mode='client', even
	 * when the tunnel later fails, so a failed reconnect does not ask again.
	 */
	get needsCode(): boolean {
		return IS_WEB_ONLY && this.mode === RemoteAccessMode.OFF;
	}

	/**
	 * A web-only build cannot render the app before the tunnel carries its
	 * requests, so the splash stays up while a restored session connects and
	 * when it fails, where it offers another code.
	 */
	get isBlocked(): boolean {
		return IS_WEB_ONLY && this.status !== TunnelStatus.CONNECTED;
	}

	async joinAsClient(shareCode: string): Promise<void> {
		if (shareCode.length < CODE_LENGTHS.SHARE) {
			throw new Error(`Invalid code: must be ${CODE_LENGTHS.SHARE} characters`);
		}

		const roomCode = shareCode.slice(0, CODE_LENGTHS.ROOM);
		const passCode = shareCode.slice(CODE_LENGTHS.ROOM);

		await this.activateClient(roomCode, passCode);
	}

	private async activateClient(
		roomCode: string,
		passCode: string,
		keepOnError = false
	): Promise<void> {
		this.mode = RemoteAccessMode.CLIENT;
		this.status = TunnelStatus.CONNECTING;
		this.errorMessage = '';
		// Install the fetch interceptor synchronously (before any await) so that
		// requests fired by layout effects on the same tick are already captured.
		this.installInterceptor();

		const tunnel = new ClientTunnel(roomCode, passCode, {
			onConnected: () => {
				this.status = TunnelStatus.CONNECTED;
			},
			onDisconnected: () => {
				this.status = TunnelStatus.ERROR;
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
			this.status = TunnelStatus.ERROR;
			this.errorMessage = e instanceof Error ? e.message : String(e);
			// Reject queued requests.
			const waiters = this.connectionWaiters.splice(0);
			const err = e instanceof Error ? e : new Error(String(e));

			for (const w of waiters) w.reject(err);

			// A restored session stays in client mode with the interceptor in
			// place, so requests fail loudly instead of silently reaching the
			// server that happens to serve this page.
			if (!keepOnError) {
				this.mode = RemoteAccessMode.OFF;
				this.uninstallInterceptor();
			}

			throw e;
		}
	}

	/** Retry the saved session after a failed or dropped connection. */
	async reconnect(): Promise<void> {
		const raw = localStorage.getItem(SESSION_KEY);

		if (!raw) return;

		const session = JSON.parse(raw) as SessionData;

		this.clientTunnel?.disconnect();
		this.clientTunnel = null;

		await this.activateClient(session.roomCode, session.passCode, true);
	}

	leaveAsClient(): void {
		this.uninstallInterceptor();
		this.clientTunnel?.disconnect();
		this.clientTunnel = null;
		this.mode = RemoteAccessMode.OFF;
		this.status = TunnelStatus.IDLE;
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
		if (this.mode === RemoteAccessMode.CLIENT && this.status === TunnelStatus.CONNECTING) {
			return new Promise<void>((resolve, reject) => {
				this.connectionWaiters.push({ reject, resolve });
			}).then(() => this.clientTunnel!.fetch(input, init));
		}

		throw new Error(
			'Remote access is enabled but the tunnel is not connected. ' +
				'Requests are not sent to the server hosting this page. ' +
				'Reconnect or leave remote access in settings.'
		);
	}

	// -------------------------------------------------------------------------
	// Persistence helpers
	// -------------------------------------------------------------------------

	private restoreSession(): void {
		try {
			const raw = localStorage.getItem(SESSION_KEY);

			if (!raw) return;

			const session = JSON.parse(raw) as SessionData;

			void this.activateClient(session.roomCode, session.passCode, true).catch(() => {
				// state is already reflected in status and errorMessage
			});
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
