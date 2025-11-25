import { browser } from '$app/environment';
import { SERVER_PROPS_LOCALSTORAGE_KEY } from '$lib/constants/localstorage-keys';
import { PropsService } from '$lib/services/props';
import { ServerMode, ModelModality } from '$lib/enums';

/**
 * PropsStore - Server properties management and mode detection
 *
 * This store manages the server properties fetched from the `/props` endpoint.
 * It provides reactive state for server configuration, capabilities, and mode detection.
 *
 * **Architecture & Relationships:**
 * - **PropsService**: Stateless service for fetching `/props` data
 * - **PropsStore** (this class): Reactive store for server properties
 * - **ModelsStore**: Uses server mode for model management strategy
 *
 * **Key Features:**
 * - **Server Properties**: Model info, context size, build information
 * - **Mode Detection**: MODEL (single model) vs ROUTER (multi-model)
 * - **Capability Detection**: Vision and audio modality support
 * - **Error Handling**: Graceful degradation with cached values
 * - **Persistence**: LocalStorage caching for offline support
 */
class PropsStore {
	constructor() {
		if (!browser) return;

		const cachedProps = this.readCachedServerProps();
		if (cachedProps) {
			this._serverProps = cachedProps;
			this.detectServerMode(cachedProps);
		}
	}

	private _serverProps = $state<ApiLlamaCppServerProps | null>(null);
	private _loading = $state(false);
	private _error = $state<string | null>(null);
	private _serverWarning = $state<string | null>(null);
	private _serverMode = $state<ServerMode | null>(null);
	private fetchPromise: Promise<void> | null = null;

	// ─────────────────────────────────────────────────────────────────────────────
	// LocalStorage persistence
	// ─────────────────────────────────────────────────────────────────────────────

	private readCachedServerProps(): ApiLlamaCppServerProps | null {
		if (!browser) return null;

		try {
			const raw = localStorage.getItem(SERVER_PROPS_LOCALSTORAGE_KEY);
			if (!raw) return null;

			return JSON.parse(raw) as ApiLlamaCppServerProps;
		} catch (error) {
			console.warn('Failed to read cached server props from localStorage:', error);
			return null;
		}
	}

	private persistServerProps(props: ApiLlamaCppServerProps | null): void {
		if (!browser) return;

		try {
			if (props) {
				localStorage.setItem(SERVER_PROPS_LOCALSTORAGE_KEY, JSON.stringify(props));
			} else {
				localStorage.removeItem(SERVER_PROPS_LOCALSTORAGE_KEY);
			}
		} catch (error) {
			console.warn('Failed to persist server props to localStorage:', error);
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Getters - Server Properties
	// ─────────────────────────────────────────────────────────────────────────────

	get serverProps(): ApiLlamaCppServerProps | null {
		return this._serverProps;
	}

	get loading(): boolean {
		return this._loading;
	}

	get error(): string | null {
		return this._error;
	}

	get serverWarning(): string | null {
		return this._serverWarning;
	}

	/**
	 * Get model name from server props.
	 * In MODEL mode: extracts from model_path or model_alias
	 * In ROUTER mode: returns null (model is per-conversation)
	 */
	get modelName(): string | null {
		if (this._serverMode === ServerMode.ROUTER) {
			return null;
		}

		if (this._serverProps?.model_alias) {
			return this._serverProps.model_alias;
		}

		if (!this._serverProps?.model_path) return null;
		return this._serverProps.model_path.split(/(\\|\/)/).pop() || null;
	}

	get supportedModalities(): ModelModality[] {
		const modalities: ModelModality[] = [];
		if (this._serverProps?.modalities?.audio) {
			modalities.push(ModelModality.AUDIO);
		}
		if (this._serverProps?.modalities?.vision) {
			modalities.push(ModelModality.VISION);
		}
		return modalities;
	}

	get supportsVision(): boolean {
		return this._serverProps?.modalities?.vision ?? false;
	}

	get supportsAudio(): boolean {
		return this._serverProps?.modalities?.audio ?? false;
	}

	get defaultParams(): ApiLlamaCppServerProps['default_generation_settings']['params'] | null {
		return this._serverProps?.default_generation_settings?.params || null;
	}

	/**
	 * Get context size (n_ctx) from server props
	 */
	get contextSize(): number | null {
		return this._serverProps?.default_generation_settings?.n_ctx ?? null;
	}

	/**
	 * Check if slots endpoint is available (set by --slots flag on server)
	 */
	get slotsEndpointAvailable(): boolean {
		return this._serverProps?.endpoint_slots ?? false;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Getters - Server Mode
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Get current server mode
	 */
	get serverMode(): ServerMode | null {
		return this._serverMode;
	}

	/**
	 * Detect if server is running in router mode (multi-model management)
	 */
	get isRouterMode(): boolean {
		return this._serverMode === ServerMode.ROUTER;
	}

	/**
	 * Detect if server is running in model mode (single model loaded)
	 */
	get isModelMode(): boolean {
		return this._serverMode === ServerMode.MODEL;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Server Mode Detection
	// ─────────────────────────────────────────────────────────────────────────────

	private detectServerMode(props: ApiLlamaCppServerProps): void {
		const newMode = props.model_path === 'none' ? ServerMode.ROUTER : ServerMode.MODEL;

		// Only log when mode changes
		if (this._serverMode !== newMode) {
			this._serverMode = newMode;
			console.info(`Server running in ${newMode === ServerMode.ROUTER ? 'ROUTER' : 'MODEL'} mode`);
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Fetch Server Properties
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Fetches server properties from the server
	 */
	async fetch(options: { silent?: boolean } = {}): Promise<void> {
		const { silent = false } = options;
		const isSilent = silent && this._serverProps !== null;

		if (this.fetchPromise) {
			return this.fetchPromise;
		}

		if (!isSilent) {
			this._loading = true;
			this._error = null;
			this._serverWarning = null;
		}

		const hadProps = this._serverProps !== null;

		const fetchPromise = (async () => {
			try {
				const props = await PropsService.fetch();
				this._serverProps = props;
				this.persistServerProps(props);
				this._error = null;
				this._serverWarning = null;

				this.detectServerMode(props);
			} catch (error) {
				if (isSilent && hadProps) {
					console.warn('Silent server props refresh failed, keeping cached data:', error);
					return;
				}

				this.handleFetchError(error, hadProps);
			} finally {
				if (!isSilent) {
					this._loading = false;
				}

				this.fetchPromise = null;
			}
		})();

		this.fetchPromise = fetchPromise;

		await fetchPromise;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Error Handling
	// ─────────────────────────────────────────────────────────────────────────────

	private handleFetchError(error: unknown, hadProps: boolean): void {
		const { errorMessage, isOfflineLikeError, isServerSideError } = this.normalizeFetchError(error);

		let cachedProps: ApiLlamaCppServerProps | null = null;

		if (!hadProps) {
			cachedProps = this.readCachedServerProps();

			if (cachedProps) {
				this._serverProps = cachedProps;
				this.detectServerMode(cachedProps);
				this._error = null;

				if (isOfflineLikeError || isServerSideError) {
					this._serverWarning = errorMessage;
				}

				console.warn(
					'Failed to refresh server properties, using cached values from localStorage:',
					errorMessage
				);
			} else {
				this._error = errorMessage;
			}
		} else {
			this._error = null;

			if (isOfflineLikeError || isServerSideError) {
				this._serverWarning = errorMessage;
			}

			console.warn(
				'Failed to refresh server properties, continuing with cached values:',
				errorMessage
			);
		}

		console.error('Error fetching server properties:', error);
	}

	private normalizeFetchError(error: unknown): {
		errorMessage: string;
		isOfflineLikeError: boolean;
		isServerSideError: boolean;
	} {
		let errorMessage = 'Failed to connect to server';
		let isOfflineLikeError = false;
		let isServerSideError = false;

		if (error instanceof Error) {
			const message = error.message || '';

			if (error.name === 'TypeError' && message.includes('fetch')) {
				errorMessage = 'Server is not running or unreachable';
				isOfflineLikeError = true;
			} else if (message.includes('ECONNREFUSED')) {
				errorMessage = 'Connection refused - server may be offline';
				isOfflineLikeError = true;
			} else if (message.includes('ENOTFOUND')) {
				errorMessage = 'Server not found - check server address';
				isOfflineLikeError = true;
			} else if (message.includes('ETIMEDOUT')) {
				errorMessage = 'Request timed out - the server took too long to respond';
				isOfflineLikeError = true;
			} else if (message.includes('503')) {
				errorMessage = 'Server temporarily unavailable - try again shortly';
				isServerSideError = true;
			} else if (message.includes('500')) {
				errorMessage = 'Server error - check server logs';
				isServerSideError = true;
			} else if (message.includes('404')) {
				errorMessage = 'Server endpoint not found';
			} else if (message.includes('403') || message.includes('401')) {
				errorMessage = 'Access denied';
			}
		}

		return { errorMessage, isOfflineLikeError, isServerSideError };
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Clear State
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Clears all server state
	 */
	clear(): void {
		this._serverProps = null;
		this._error = null;
		this._serverWarning = null;
		this._loading = false;
		this._serverMode = null;
		this.fetchPromise = null;
		this.persistServerProps(null);
	}
}

export const propsStore = new PropsStore();

// ─────────────────────────────────────────────────────────────────────────────
// Reactive Getters (for use in components)
// ─────────────────────────────────────────────────────────────────────────────

export const serverProps = () => propsStore.serverProps;
export const propsLoading = () => propsStore.loading;
export const propsError = () => propsStore.error;
export const serverWarning = () => propsStore.serverWarning;
export const modelName = () => propsStore.modelName;
export const supportedModalities = () => propsStore.supportedModalities;
export const supportsVision = () => propsStore.supportsVision;
export const supportsAudio = () => propsStore.supportsAudio;
export const slotsEndpointAvailable = () => propsStore.slotsEndpointAvailable;
export const defaultParams = () => propsStore.defaultParams;
export const contextSize = () => propsStore.contextSize;

// Server mode exports
export const serverMode = () => propsStore.serverMode;
export const isRouterMode = () => propsStore.isRouterMode;
export const isModelMode = () => propsStore.isModelMode;

// Actions
export const fetchProps = propsStore.fetch.bind(propsStore);
