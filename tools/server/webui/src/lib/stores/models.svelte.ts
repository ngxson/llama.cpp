import { SvelteSet } from 'svelte/reactivity';
import { ModelsService } from '$lib/services/models';
import { ServerModelStatus } from '$lib/enums';
import type { ModelOption } from '$lib/types/models';
import type { ApiRouterModelMeta } from '$lib/types/api';

/**
 * ModelsStore - Reactive store for model management in both MODEL and ROUTER modes
 *
 * This store manages:
 * - Available models list
 * - Selected model for new conversations
 * - Loaded models tracking (ROUTER mode)
 * - Model usage tracking per conversation
 * - Automatic unloading of unused models
 *
 * **Architecture & Relationships:**
 * - **ModelsService**: Stateless service for API communication
 * - **ModelsStore** (this class): Reactive store for model state
 * - **PropsStore**: Provides server mode detection
 * - **ConversationsStore**: Tracks which conversations use which models
 *
 * **Key Features:**
 * - **MODEL mode**: Single model, always loaded
 * - **ROUTER mode**: Multi-model with load/unload capability
 * - **Auto-unload**: Automatically unloads models not used by any conversation
 * - **Lazy loading**: ensureModelLoaded() loads models on demand
 */
class ModelsStore {
	// ─────────────────────────────────────────────────────────────────────────────
	// State
	// ─────────────────────────────────────────────────────────────────────────────

	private _models = $state<ModelOption[]>([]);
	private _routerModels = $state<ApiRouterModelMeta[]>([]);
	private _loading = $state(false);
	private _updating = $state(false);
	private _error = $state<string | null>(null);
	private _selectedModelId = $state<string | null>(null);
	private _selectedModelName = $state<string | null>(null);

	/** Maps modelId -> Set of conversationIds that use this model */
	private _modelUsage = $state<Map<string, SvelteSet<string>>>(new Map());

	/** Maps modelId -> loading state for load/unload operations */
	private _modelLoadingStates = $state<Map<string, boolean>>(new Map());

	// ─────────────────────────────────────────────────────────────────────────────
	// Getters - Basic
	// ─────────────────────────────────────────────────────────────────────────────

	get models(): ModelOption[] {
		return this._models;
	}

	get routerModels(): ApiRouterModelMeta[] {
		return this._routerModels;
	}

	get loading(): boolean {
		return this._loading;
	}

	get updating(): boolean {
		return this._updating;
	}

	get error(): string | null {
		return this._error;
	}

	get selectedModelId(): string | null {
		return this._selectedModelId;
	}

	get selectedModelName(): string | null {
		return this._selectedModelName;
	}

	get selectedModel(): ModelOption | null {
		if (!this._selectedModelId) {
			return null;
		}

		return this._models.find((model) => model.id === this._selectedModelId) ?? null;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Getters - Loaded Models (ROUTER mode)
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Get list of currently loaded model IDs
	 */
	get loadedModelIds(): string[] {
		return this._routerModels
			.filter((m) => m.status === ServerModelStatus.LOADED)
			.map((m) => m.name);
	}

	/**
	 * Get list of models currently being loaded/unloaded
	 */
	get loadingModelIds(): string[] {
		return Array.from(this._modelLoadingStates.entries())
			.filter(([, loading]) => loading)
			.map(([id]) => id);
	}

	/**
	 * Check if a specific model is loaded
	 */
	isModelLoaded(modelId: string): boolean {
		const model = this._routerModels.find((m) => m.name === modelId);
		return model?.status === ServerModelStatus.LOADED || false;
	}

	/**
	 * Check if a specific model is currently loading/unloading
	 */
	isModelOperationInProgress(modelId: string): boolean {
		return this._modelLoadingStates.get(modelId) ?? false;
	}

	/**
	 * Get the status of a specific model
	 */
	getModelStatus(modelId: string): ServerModelStatus | null {
		const model = this._routerModels.find((m) => m.name === modelId);
		return model?.status ?? null;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Getters - Model Usage
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Get set of conversation IDs using a specific model
	 */
	getModelUsage(modelId: string): SvelteSet<string> {
		return this._modelUsage.get(modelId) ?? new SvelteSet<string>();
	}

	/**
	 * Check if a model is used by any conversation
	 */
	isModelInUse(modelId: string): boolean {
		const usage = this._modelUsage.get(modelId);
		return usage !== undefined && usage.size > 0;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Fetch Models
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Fetch list of models from server
	 */
	async fetch(force = false): Promise<void> {
		if (this._loading) return;
		if (this._models.length > 0 && !force) return;

		this._loading = true;
		this._error = null;

		try {
			const response = await ModelsService.list();

			const models: ModelOption[] = response.data.map((item, index) => {
				const details = response.models?.[index];
				const rawCapabilities = Array.isArray(details?.capabilities) ? details?.capabilities : [];
				const displayNameSource =
					details?.name && details.name.trim().length > 0 ? details.name : item.id;
				const displayName = this.toDisplayName(displayNameSource);

				return {
					id: item.id,
					name: displayName,
					model: details?.model || item.id,
					description: details?.description,
					capabilities: rawCapabilities.filter((value): value is string => Boolean(value)),
					details: details?.details,
					meta: item.meta ?? null
				} satisfies ModelOption;
			});

			this._models = models;

			// Don't auto-select any model - selection should come from:
			// 1. User explicitly selecting a model in the UI
			// 2. Conversation model (synced via ChatFormActions effect)
		} catch (error) {
			this._models = [];
			this._error = error instanceof Error ? error.message : 'Failed to load models';

			throw error;
		} finally {
			this._loading = false;
		}
	}

	/**
	 * Fetch router models with full metadata (ROUTER mode only)
	 */
	async fetchRouterModels(): Promise<void> {
		try {
			const response = await ModelsService.listRouter();
			this._routerModels = response.models;
		} catch (error) {
			console.warn('Failed to fetch router models:', error);
			this._routerModels = [];
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Select Model
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Select a model for new conversations
	 */
	async select(modelId: string): Promise<void> {
		if (!modelId || this._updating) {
			return;
		}

		if (this._selectedModelId === modelId) {
			return;
		}

		const option = this._models.find((model) => model.id === modelId);
		if (!option) {
			throw new Error('Selected model is not available');
		}

		this._updating = true;
		this._error = null;

		try {
			this._selectedModelId = option.id;
			this._selectedModelName = option.model;
		} finally {
			this._updating = false;
		}
	}

	/**
	 * Select a model by its model name (used for syncing with conversation model)
	 * @param modelName - Model name to select (e.g., "unsloth/gemma-3-12b-it-GGUF:latest")
	 */
	selectModelByName(modelName: string): void {
		const option = this._models.find((model) => model.model === modelName);
		if (option) {
			this._selectedModelId = option.id;
			this._selectedModelName = option.model;
			// Don't persist - this is just for syncing with conversation
		}
	}

	/**
	 * Clear the current model selection
	 */
	clearSelection(): void {
		this._selectedModelId = null;
		this._selectedModelName = null;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Load/Unload Models (ROUTER mode)
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Load a model (ROUTER mode)
	 * @param modelId - Model identifier to load
	 */
	async loadModel(modelId: string): Promise<void> {
		if (this.isModelLoaded(modelId)) {
			return;
		}

		if (this._modelLoadingStates.get(modelId)) {
			return; // Already loading
		}

		this._modelLoadingStates.set(modelId, true);
		this._error = null;

		try {
			await ModelsService.load(modelId);
			await this.fetchRouterModels(); // Refresh status
		} catch (error) {
			this._error = error instanceof Error ? error.message : 'Failed to load model';
			throw error;
		} finally {
			this._modelLoadingStates.set(modelId, false);
		}
	}

	/**
	 * Unload a model (ROUTER mode)
	 * @param modelId - Model identifier to unload
	 */
	async unloadModel(modelId: string): Promise<void> {
		if (!this.isModelLoaded(modelId)) {
			return;
		}

		if (this._modelLoadingStates.get(modelId)) {
			return; // Already unloading
		}

		this._modelLoadingStates.set(modelId, true);
		this._error = null;

		try {
			await ModelsService.unload(modelId);
			await this.fetchRouterModels(); // Refresh status
		} catch (error) {
			this._error = error instanceof Error ? error.message : 'Failed to unload model';
			throw error;
		} finally {
			this._modelLoadingStates.set(modelId, false);
		}
	}

	/**
	 * Ensure a model is loaded before use
	 * @param modelId - Model identifier to ensure is loaded
	 */
	async ensureModelLoaded(modelId: string): Promise<void> {
		if (this.isModelLoaded(modelId)) {
			return;
		}

		await this.loadModel(modelId);
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Model Usage Tracking
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Register that a conversation is using a model
	 */
	registerModelUsage(modelId: string, conversationId: string): void {
		const usage = this._modelUsage.get(modelId) ?? new SvelteSet<string>();
		usage.add(conversationId);
		this._modelUsage.set(modelId, usage);
	}

	/**
	 * Unregister that a conversation is using a model
	 * @param modelId - Model identifier
	 * @param conversationId - Conversation identifier
	 * @param autoUnload - Whether to automatically unload the model if no longer used
	 */
	async unregisterModelUsage(
		modelId: string,
		conversationId: string,
		autoUnload = true
	): Promise<void> {
		const usage = this._modelUsage.get(modelId);
		if (usage) {
			usage.delete(conversationId);

			if (usage.size === 0) {
				this._modelUsage.delete(modelId);

				// Auto-unload if model is not used by any conversation
				if (autoUnload && this.isModelLoaded(modelId)) {
					await this.unloadModel(modelId);
				}
			}
		}
	}

	/**
	 * Clear all usage for a conversation (when conversation is deleted)
	 */
	async clearConversationUsage(conversationId: string): Promise<void> {
		for (const [modelId, usage] of this._modelUsage.entries()) {
			if (usage.has(conversationId)) {
				await this.unregisterModelUsage(modelId, conversationId);
			}
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Private Helpers
	// ─────────────────────────────────────────────────────────────────────────────

	private toDisplayName(id: string): string {
		const segments = id.split(/\\|\//);
		const candidate = segments.pop();

		return candidate && candidate.trim().length > 0 ? candidate : id;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Clear State
	// ─────────────────────────────────────────────────────────────────────────────

	clear(): void {
		this._models = [];
		this._routerModels = [];
		this._loading = false;
		this._updating = false;
		this._error = null;
		this._selectedModelId = null;
		this._selectedModelName = null;
		this._modelUsage.clear();
		this._modelLoadingStates.clear();
	}
}

export const modelsStore = new ModelsStore();

// ─────────────────────────────────────────────────────────────────────────────
// Reactive Getters
// ─────────────────────────────────────────────────────────────────────────────

export const modelOptions = () => modelsStore.models;
export const routerModels = () => modelsStore.routerModels;
export const modelsLoading = () => modelsStore.loading;
export const modelsUpdating = () => modelsStore.updating;
export const modelsError = () => modelsStore.error;
export const selectedModelId = () => modelsStore.selectedModelId;
export const selectedModelName = () => modelsStore.selectedModelName;
export const selectedModelOption = () => modelsStore.selectedModel;
export const loadedModelIds = () => modelsStore.loadedModelIds;
export const loadingModelIds = () => modelsStore.loadingModelIds;

// ─────────────────────────────────────────────────────────────────────────────
// Actions
// ─────────────────────────────────────────────────────────────────────────────

export const fetchModels = modelsStore.fetch.bind(modelsStore);
export const fetchRouterModels = modelsStore.fetchRouterModels.bind(modelsStore);
export const selectModel = modelsStore.select.bind(modelsStore);
export const loadModel = modelsStore.loadModel.bind(modelsStore);
export const unloadModel = modelsStore.unloadModel.bind(modelsStore);
export const ensureModelLoaded = modelsStore.ensureModelLoaded.bind(modelsStore);
export const registerModelUsage = modelsStore.registerModelUsage.bind(modelsStore);
export const unregisterModelUsage = modelsStore.unregisterModelUsage.bind(modelsStore);
export const clearConversationUsage = modelsStore.clearConversationUsage.bind(modelsStore);
export const selectModelByName = modelsStore.selectModelByName.bind(modelsStore);
export const clearModelSelection = modelsStore.clearSelection.bind(modelsStore);
