<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { ChevronDown, Loader2, Package } from '@lucide/svelte';
	import { cn } from '$lib/components/ui/utils';
	import { portalToBody } from '$lib/utils/portal-to-body';
	import {
		fetchModels,
		modelOptions,
		modelsLoading,
		modelsUpdating,
		selectModel,
		selectedModelId
	} from '$lib/stores/models.svelte';
	import { isRouterMode, serverStore } from '$lib/stores/server.svelte';
	import { DialogModelInformation } from '$lib/components/app';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		class?: string;
		currentModel?: string | null;
		onModelChange?: (modelId: string, modelName: string) => void;
		disabled?: boolean;
		forceForegroundText?: boolean;
	}

	let {
		class: className = '',
		currentModel = null,
		onModelChange,
		disabled = false,
		forceForegroundText = false
	}: Props = $props();

	let options = $derived(modelOptions());
	let loading = $derived(modelsLoading());
	let updating = $derived(modelsUpdating());
	let activeId = $derived(selectedModelId());
	let isRouter = $derived(isRouterMode());
	let serverModel = $derived(serverStore.modelName);

	let isHighlightedCurrentModelActive = $derived(
		!isRouter || !currentModel
			? false
			: (() => {
					const currentOption = options.find((option) => option.model === currentModel);

					return currentOption ? currentOption.id === activeId : false;
				})()
	);

	let isCurrentModelInCache = $derived(() => {
		if (!isRouter || !currentModel) return true;

		return options.some((option) => option.model === currentModel);
	});

	let isOpen = $state(false);
	let showModelDialog = $state(false);
	let container: HTMLDivElement | null = null;
	let menuRef = $state<HTMLDivElement | null>(null);
	let triggerButton = $state<HTMLButtonElement | null>(null);
	let menuPosition = $state<{
		top: number;
		left: number;
		width: number;
		placement: 'top' | 'bottom';
		maxHeight: number;
	} | null>(null);

	const VIEWPORT_GUTTER = 8;
	const MENU_OFFSET = 6;
	const MENU_MAX_WIDTH = 320;

	onMount(async () => {
		try {
			await fetchModels();
		} catch (error) {
			console.error('Unable to load models:', error);
		}
	});

	function toggleOpen() {
		if (loading || updating) return;

		if (isRouter) {
			// Router mode: show dropdown
			if (isOpen) {
				closeMenu();
			} else {
				openMenu();
			}
		} else {
			// Single model mode: show dialog
			showModelDialog = true;
		}
	}

	async function openMenu() {
		if (loading || updating) return;

		isOpen = true;
		await tick();
		updateMenuPosition();
		requestAnimationFrame(() => updateMenuPosition());
	}

	function closeMenu() {
		if (!isOpen) return;

		isOpen = false;
		menuPosition = null;
	}

	function handlePointerDown(event: PointerEvent) {
		if (!container) return;

		const target = event.target as Node | null;

		if (target && !container.contains(target) && !(menuRef && menuRef.contains(target))) {
			closeMenu();
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			closeMenu();
		}
	}

	function handleResize() {
		if (isOpen) {
			updateMenuPosition();
		}
	}

	function updateMenuPosition() {
		if (!isOpen || !triggerButton || !menuRef) return;

		const triggerRect = triggerButton.getBoundingClientRect();
		const viewportWidth = window.innerWidth;
		const viewportHeight = window.innerHeight;

		if (viewportWidth === 0 || viewportHeight === 0) return;

		const scrollWidth = menuRef.scrollWidth;
		const scrollHeight = menuRef.scrollHeight;

		const availableWidth = Math.max(0, viewportWidth - VIEWPORT_GUTTER * 2);
		const constrainedMaxWidth = Math.min(MENU_MAX_WIDTH, availableWidth || MENU_MAX_WIDTH);
		const safeMaxWidth =
			constrainedMaxWidth > 0 ? constrainedMaxWidth : Math.min(MENU_MAX_WIDTH, viewportWidth);
		const desiredMinWidth = Math.min(160, safeMaxWidth || 160);

		let width = Math.min(
			Math.max(triggerRect.width, scrollWidth, desiredMinWidth),
			safeMaxWidth || 320
		);

		const availableBelow = Math.max(
			0,
			viewportHeight - VIEWPORT_GUTTER - triggerRect.bottom - MENU_OFFSET
		);
		const availableAbove = Math.max(0, triggerRect.top - VIEWPORT_GUTTER - MENU_OFFSET);
		const viewportAllowance = Math.max(0, viewportHeight - VIEWPORT_GUTTER * 2);
		const fallbackAllowance = Math.max(1, viewportAllowance > 0 ? viewportAllowance : scrollHeight);

		function computePlacement(placement: 'top' | 'bottom') {
			const available = placement === 'bottom' ? availableBelow : availableAbove;
			const allowedHeight =
				available > 0 ? Math.min(available, fallbackAllowance) : fallbackAllowance;
			const maxHeight = Math.min(scrollHeight, allowedHeight);
			const height = Math.max(0, maxHeight);

			let top: number;
			if (placement === 'bottom') {
				const rawTop = triggerRect.bottom + MENU_OFFSET;
				const minTop = VIEWPORT_GUTTER;
				const maxTop = viewportHeight - VIEWPORT_GUTTER - height;
				if (maxTop < minTop) {
					top = minTop;
				} else {
					top = Math.min(Math.max(rawTop, minTop), maxTop);
				}
			} else {
				const rawTop = triggerRect.top - MENU_OFFSET - height;
				const minTop = VIEWPORT_GUTTER;
				const maxTop = viewportHeight - VIEWPORT_GUTTER - height;
				if (maxTop < minTop) {
					top = minTop;
				} else {
					top = Math.max(Math.min(rawTop, maxTop), minTop);
				}
			}

			return { placement, top, height, maxHeight };
		}

		const belowMetrics = computePlacement('bottom');
		const aboveMetrics = computePlacement('top');

		let metrics = belowMetrics;
		if (scrollHeight > belowMetrics.maxHeight && aboveMetrics.maxHeight > belowMetrics.maxHeight) {
			metrics = aboveMetrics;
		}

		let left = triggerRect.right - width;
		const maxLeft = viewportWidth - VIEWPORT_GUTTER - width;
		if (maxLeft < VIEWPORT_GUTTER) {
			left = VIEWPORT_GUTTER;
		} else {
			if (left > maxLeft) {
				left = maxLeft;
			}
			if (left < VIEWPORT_GUTTER) {
				left = VIEWPORT_GUTTER;
			}
		}

		menuPosition = {
			top: Math.round(metrics.top),
			left: Math.round(left),
			width: Math.round(width),
			placement: metrics.placement,
			maxHeight: Math.round(metrics.maxHeight)
		};
	}

	function handleSelect(modelId: string) {
		const option = options.find((opt) => opt.id === modelId);
		if (option && onModelChange) {
			// If callback provided, use it (for regenerate functionality)
			onModelChange(option.id, option.model);
		} else if (option) {
			// Otherwise, just update the global selection (for form selector)
			selectModel(option.id).catch(console.error);
		}
		closeMenu();
	}

	function getDisplayOption(): ModelOption | undefined {
		if (!isRouter) {
			if (serverModel) {
				return {
					id: 'current',
					model: serverModel,
					name: serverModel.split('/').pop() || serverModel,
					capabilities: [] // Empty array for single model mode
				};
			}

			return undefined;
		}

		if (currentModel) {
			if (!isCurrentModelInCache()) {
				return {
					id: 'not-in-cache',
					model: currentModel,
					name: currentModel.split('/').pop() || currentModel,
					capabilities: []
				};
			}

			return options.find((option) => option.model === currentModel);
		}

		if (activeId) {
			return options.find((option) => option.id === activeId);
		}

		return options[0];
	}
</script>

<svelte:window onresize={handleResize} />
<svelte:document onpointerdown={handlePointerDown} onkeydown={handleKeydown} />

<div class={cn('relative inline-flex flex-col items-end gap-1', className)} bind:this={container}>
	{#if loading && options.length === 0 && isRouter}
		<div class="flex items-center gap-2 text-xs text-muted-foreground">
			<Loader2 class="h-3.5 w-3.5 animate-spin" />
			Loading models…
		</div>
	{:else if options.length === 0 && isRouter}
		<p class="text-xs text-muted-foreground">No models available.</p>
	{:else}
		{@const selectedOption = getDisplayOption()}

		<div class="relative">
			<button
				type="button"
				class={cn(
					`inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60`,
					!isCurrentModelInCache()
						? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
						: forceForegroundText
							? 'text-foreground'
							: isHighlightedCurrentModelActive
								? 'text-foreground'
								: 'text-muted-foreground',
					isOpen ? 'text-foreground' : '',
					className
				)}
				style="max-width: min(calc(100cqw - 6.5rem), 32rem)"
				aria-haspopup={isRouter ? 'listbox' : undefined}
				aria-expanded={isRouter ? isOpen : undefined}
				onclick={toggleOpen}
				bind:this={triggerButton}
				disabled={disabled || updating}
			>
				<Package class="h-3.5 w-3.5" />

				<span class="truncate font-medium">
					{selectedOption?.model || 'Select model'}
				</span>

				{#if updating}
					<Loader2 class="h-3 w-3.5 animate-spin" />
				{:else if isRouter}
					<ChevronDown class="h-3 w-3.5" />
				{/if}
			</button>

			{#if isOpen && isRouter}
				<div
					bind:this={menuRef}
					use:portalToBody
					class={cn(
						'fixed z-[1000] overflow-hidden rounded-md border bg-popover shadow-lg transition-opacity',
						menuPosition ? 'opacity-100' : 'pointer-events-none opacity-0'
					)}
					role="listbox"
					style:top={menuPosition ? `${menuPosition.top}px` : undefined}
					style:left={menuPosition ? `${menuPosition.left}px` : undefined}
					style:width={menuPosition ? `${menuPosition.width}px` : undefined}
					data-placement={menuPosition?.placement ?? 'bottom'}
				>
					<div
						class="overflow-y-auto py-1"
						style:max-height={menuPosition && menuPosition.maxHeight > 0
							? `${menuPosition.maxHeight}px`
							: undefined}
					>
						{#if !isCurrentModelInCache() && currentModel}
							<!-- Show unavailable model as first option (disabled) -->
							<button
								type="button"
								class="flex w-full cursor-not-allowed items-center bg-red-400/10 px-3 py-2 text-left text-sm text-red-400"
								role="option"
								aria-selected="true"
								aria-disabled="true"
								disabled
							>
								<span class="truncate">{selectedOption?.name || currentModel}</span>
								<span class="ml-2 text-xs whitespace-nowrap opacity-70">(not available)</span>
							</button>
							<div class="my-1 h-px bg-border"></div>
						{/if}
						{#each options as option (option.id)}
							<button
								type="button"
								class={cn(
									'flex w-full cursor-pointer items-center px-3 py-2 text-left text-sm transition hover:bg-muted focus:bg-muted focus:outline-none',
									currentModel === option.model || activeId === option.id
										? 'bg-accent text-accent-foreground'
										: 'text-popover-foreground hover:bg-accent hover:text-accent-foreground'
								)}
								role="option"
								aria-selected={currentModel === option.model || activeId === option.id}
								onclick={() => handleSelect(option.id)}
							>
								<span class="truncate">{option.model}</span>
							</button>
						{/each}
					</div>
				</div>
			{/if}
		</div>
	{/if}
</div>

{#if showModelDialog && !isRouter}
	<DialogModelInformation bind:open={showModelDialog} />
{/if}
