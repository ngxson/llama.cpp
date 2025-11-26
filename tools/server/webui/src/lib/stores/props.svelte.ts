import { PropsService } from '$lib/services/props';
import { ServerRole, ModelModality } from '$lib/enums';

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
 * - **Error Handling**: Clear error states when server unavailable
 */
class PropsStore {
	private _serverProps = $state<ApiLlamaCppServerProps | null>(null);
	private _loading = $state(false);
	private _error = $state<string | null>(null);
	private _serverRole = $state<ServerRole | null>(null);
	private fetchPromise: Promise<void> | null = null;

	// Model-specific props cache (ROUTER mode)
	private _modelPropsCache = $state<Map<string, ApiLlamaCppServerProps>>(new Map());
	private _modelPropsFetching = $state<Set<string>>(new Set());

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

	/**
	 * Get model name from server props.
	 * In MODEL mode: extracts from model_path or model_alias
	 * In ROUTER mode: returns null (model is per-conversation)
	 */
	get modelName(): string | null {
		if (this._serverRole === ServerRole.ROUTER) {
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
	get serverRole(): ServerRole | null {
		return this._serverRole;
	}

	/**
	 * Detect if server is running in router mode (multi-model management)
	 */
	get isRouterMode(): boolean {
		return this._serverRole === ServerRole.ROUTER;
	}

	/**
	 * Detect if server is running in model mode (single model loaded)
	 */
	get isModelMode(): boolean {
		return this._serverRole === ServerRole.MODEL;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Server Mode Detection
	// ─────────────────────────────────────────────────────────────────────────────

	private detectServerRole(props: ApiLlamaCppServerProps): void {
		console.log('Server props role:', props?.role);
		const newMode =
			// todo - `role` attribute should always be available on the `/props` endpoint
			props?.role === ServerRole.ROUTER ? ServerRole.ROUTER : ServerRole.MODEL;

		// Only log when mode changes
		if (this._serverRole !== newMode) {
			this._serverRole = newMode;
			console.info(`Server running in ${newMode === ServerRole.ROUTER ? 'ROUTER' : 'MODEL'} mode`);
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Fetch Server Properties
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Fetches server properties from the server
	 */
	async fetch(): Promise<void> {
		if (this.fetchPromise) {
			return this.fetchPromise;
		}

		this._loading = true;
		this._error = null;

		const previousBuildInfo = this._serverProps?.build_info;

		const fetchPromise = (async () => {
			try {
				const props = await PropsService.fetch();

				// Clear model-specific props cache if server was restarted
				if (previousBuildInfo && previousBuildInfo !== props.build_info) {
					this._modelPropsCache.clear();
					console.info('Cleared model props cache due to server restart');
				}

				this._serverProps = props;
				this._error = null;
				this.detectServerRole(props);
			} catch (error) {
				this._error = this.getErrorMessage(error);
				console.error('Error fetching server properties:', error);
			} finally {
				this._loading = false;
				this.fetchPromise = null;
			}
		})();

		this.fetchPromise = fetchPromise;
		await fetchPromise;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Fetch Model-Specific Properties (ROUTER mode)
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Get cached props for a specific model
	 */
	getModelProps(modelId: string): ApiLlamaCppServerProps | null {
		return this._modelPropsCache.get(modelId) ?? null;
	}

	/**
	 * Check if model props are being fetched
	 */
	isModelPropsFetching(modelId: string): boolean {
		return this._modelPropsFetching.has(modelId);
	}

	/**
	 * Fetches properties for a specific model (ROUTER mode)
	 * Results are cached for subsequent calls
	 */
	async fetchModelProps(modelId: string): Promise<ApiLlamaCppServerProps | null> {
		// Return cached if available
		const cached = this._modelPropsCache.get(modelId);
		if (cached) return cached;

		// Don't fetch if already fetching
		if (this._modelPropsFetching.has(modelId)) {
			return null;
		}

		this._modelPropsFetching.add(modelId);

		try {
			const props = await PropsService.fetchForModel(modelId);
			this._modelPropsCache.set(modelId, props);
			return props;
		} catch (error) {
			console.warn(`Failed to fetch props for model ${modelId}:`, error);
			return null;
		} finally {
			this._modelPropsFetching.delete(modelId);
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Error Handling
	// ─────────────────────────────────────────────────────────────────────────────

	private getErrorMessage(error: unknown): string {
		if (error instanceof Error) {
			const message = error.message || '';

			if (error.name === 'TypeError' && message.includes('fetch')) {
				return 'Server is not running or unreachable';
			} else if (message.includes('ECONNREFUSED')) {
				return 'Connection refused - server may be offline';
			} else if (message.includes('ENOTFOUND')) {
				return 'Server not found - check server address';
			} else if (message.includes('ETIMEDOUT')) {
				return 'Request timed out';
			} else if (message.includes('503')) {
				return 'Server temporarily unavailable';
			} else if (message.includes('500')) {
				return 'Server error - check server logs';
			} else if (message.includes('404')) {
				return 'Server endpoint not found';
			} else if (message.includes('403') || message.includes('401')) {
				return 'Access denied';
			}
		}

		return 'Failed to connect to server';
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
		this._loading = false;
		this._serverRole = null;
		this.fetchPromise = null;
		this._modelPropsCache.clear();
	}
}

export const propsStore = new PropsStore();

// ─────────────────────────────────────────────────────────────────────────────
// Reactive Getters (for use in components)
// ─────────────────────────────────────────────────────────────────────────────

export const serverProps = () => propsStore.serverProps;
export const propsLoading = () => propsStore.loading;
export const propsError = () => propsStore.error;
export const modelName = () => propsStore.modelName;
export const supportedModalities = () => propsStore.supportedModalities;
export const supportsVision = () => propsStore.supportsVision;
export const supportsAudio = () => propsStore.supportsAudio;
export const slotsEndpointAvailable = () => propsStore.slotsEndpointAvailable;
export const defaultParams = () => propsStore.defaultParams;
export const contextSize = () => propsStore.contextSize;

// Server mode exports
export const serverRole = () => propsStore.serverRole;
export const isRouterMode = () => propsStore.isRouterMode;
export const isModelMode = () => propsStore.isModelMode;

// Actions
export const fetchProps = propsStore.fetch.bind(propsStore);
export const fetchModelProps = propsStore.fetchModelProps.bind(propsStore);
export const getModelProps = propsStore.getModelProps.bind(propsStore);
