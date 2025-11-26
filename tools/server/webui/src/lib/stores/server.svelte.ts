import { PropsService } from '$lib/services/props';
import { ServerRole, ModelModality } from '$lib/enums';

/**
 * ServerStore - Server connection state, configuration, and role detection
 *
 * This store manages the server connection state and properties fetched from `/props`.
 * It provides reactive state for server configuration and role detection.
 *
 * **Architecture & Relationships:**
 * - **PropsService**: Stateless service for fetching `/props` data
 * - **ServerStore** (this class): Reactive store for server state
 * - **ModelsStore**: Independent store for model management (uses PropsService directly)
 *
 * **Key Features:**
 * - **Server State**: Connection status, loading, error handling
 * - **Role Detection**: MODEL (single model) vs ROUTER (multi-model)
 * - **Default Params**: Server-wide generation defaults
 *
 * **Note on Modalities:**
 * Model-specific modalities (vision, audio) are now managed by ModelsStore.
 * Use `modelsStore.getModelModalities(modelId)` for per-model modality info.
 * The `supportsVision`/`supportsAudio` getters here are deprecated and only
 * apply to MODEL mode (single model).
 */
class ServerStore {
	props = $state<ApiLlamaCppServerProps | null>(null);
	loading = $state(false);
	error = $state<string | null>(null);
	role = $state<ServerRole | null>(null);
	private fetchPromise: Promise<void> | null = null;

	// ─────────────────────────────────────────────────────────────────────────────
	// Computed Getters
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Get model name from server props.
	 * In MODEL mode: extracts from model_path or model_alias
	 * In ROUTER mode: returns null (model is per-conversation)
	 */
	get modelName(): string | null {
		if (this.role === ServerRole.ROUTER) return null;
		if (this.props?.model_alias) return this.props.model_alias;
		if (!this.props?.model_path) return null;
		return this.props.model_path.split(/(\\|\/)/).pop() || null;
	}

	/**
	 * @deprecated Use modelsStore.getModelModalities(modelId) for per-model modalities.
	 * This only works in MODEL mode (single model).
	 */
	get supportedModalities(): ModelModality[] {
		const modalities: ModelModality[] = [];
		if (this.props?.modalities?.audio) modalities.push(ModelModality.AUDIO);
		if (this.props?.modalities?.vision) modalities.push(ModelModality.VISION);
		return modalities;
	}

	/**
	 * @deprecated Use modelsStore.modelSupportsVision(modelId) for per-model check.
	 * This only works in MODEL mode (single model).
	 */
	get supportsVision(): boolean {
		return this.props?.modalities?.vision ?? false;
	}

	/**
	 * @deprecated Use modelsStore.modelSupportsAudio(modelId) for per-model check.
	 * This only works in MODEL mode (single model).
	 */
	get supportsAudio(): boolean {
		return this.props?.modalities?.audio ?? false;
	}

	get defaultParams(): ApiLlamaCppServerProps['default_generation_settings']['params'] | null {
		return this.props?.default_generation_settings?.params || null;
	}

	get contextSize(): number | null {
		return this.props?.default_generation_settings?.n_ctx ?? null;
	}

	get slotsEndpointAvailable(): boolean {
		return this.props?.endpoint_slots ?? false;
	}

	get isRouterMode(): boolean {
		return this.role === ServerRole.ROUTER;
	}

	get isModelMode(): boolean {
		return this.role === ServerRole.MODEL;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Server Role Detection
	// ─────────────────────────────────────────────────────────────────────────────

	private detectRole(props: ApiLlamaCppServerProps): void {
		const newRole = props?.role === ServerRole.ROUTER ? ServerRole.ROUTER : ServerRole.MODEL;
		if (this.role !== newRole) {
			this.role = newRole;
			console.info(`Server running in ${newRole === ServerRole.ROUTER ? 'ROUTER' : 'MODEL'} mode`);
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Fetch Server Properties
	// ─────────────────────────────────────────────────────────────────────────────

	async fetch(): Promise<void> {
		if (this.fetchPromise) return this.fetchPromise;

		this.loading = true;
		this.error = null;

		const fetchPromise = (async () => {
			try {
				const props = await PropsService.fetch();
				this.props = props;
				this.error = null;
				this.detectRole(props);
			} catch (error) {
				this.error = this.getErrorMessage(error);
				console.error('Error fetching server properties:', error);
			} finally {
				this.loading = false;
				this.fetchPromise = null;
			}
		})();

		this.fetchPromise = fetchPromise;
		await fetchPromise;
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

	clear(): void {
		this.props = null;
		this.error = null;
		this.loading = false;
		this.role = null;
		this.fetchPromise = null;
	}
}

export const serverStore = new ServerStore();

// ─────────────────────────────────────────────────────────────────────────────
// Reactive Getters (for use in components)
// ─────────────────────────────────────────────────────────────────────────────

export const serverProps = () => serverStore.props;
export const serverLoading = () => serverStore.loading;
export const serverError = () => serverStore.error;
export const serverRole = () => serverStore.role;
export const modelName = () => serverStore.modelName;
export const supportedModalities = () => serverStore.supportedModalities;
export const supportsVision = () => serverStore.supportsVision;
export const supportsAudio = () => serverStore.supportsAudio;
export const slotsEndpointAvailable = () => serverStore.slotsEndpointAvailable;
export const defaultParams = () => serverStore.defaultParams;
export const contextSize = () => serverStore.contextSize;
export const isRouterMode = () => serverStore.isRouterMode;
export const isModelMode = () => serverStore.isModelMode;

// Actions
export const fetchServerProps = serverStore.fetch.bind(serverStore);
