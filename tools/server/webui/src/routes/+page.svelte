<script lang="ts">
	import { ChatScreen, DialogModelNotAvailable } from '$lib/components/app';
	import { sendMessage, clearUIState } from '$lib/stores/chat.svelte';
	import {
		conversationsStore,
		isConversationsInitialized,
		clearActiveConversation,
		createConversation
	} from '$lib/stores/conversations.svelte';
	import {
		fetchModels,
		modelOptions,
		selectModel,
		findModelByName
	} from '$lib/stores/models.svelte';
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { replaceState } from '$app/navigation';

	let qParam = $derived(page.url.searchParams.get('q'));
	let modelParam = $derived(page.url.searchParams.get('model'));
	let newChatParam = $derived(page.url.searchParams.get('new_chat'));

	// Dialog state for model not available error
	let showModelNotAvailable = $state(false);
	let requestedModelName = $state('');
	let availableModelNames = $derived(modelOptions().map((m) => m.model));

	/**
	 * Clear URL params after message is sent to prevent re-sending on refresh
	 */
	function clearUrlParams() {
		const url = new URL(page.url);
		url.searchParams.delete('q');
		url.searchParams.delete('model');
		url.searchParams.delete('new_chat');
		replaceState(url.toString(), {});
	}

	async function handleUrlParams() {
		// Ensure models are loaded first
		await fetchModels();

		// Handle model parameter - select model if provided
		if (modelParam) {
			const model = findModelByName(modelParam);
			if (model) {
				try {
					await selectModel(model.id);
				} catch (error) {
					console.error('Failed to select model:', error);
					requestedModelName = modelParam;
					showModelNotAvailable = true;
					return;
				}
			} else {
				// Model not found - show error dialog
				requestedModelName = modelParam;
				showModelNotAvailable = true;
				return;
			}
		}

		// Handle ?q= parameter - create new conversation and send message
		if (qParam !== null) {
			await createConversation();
			await sendMessage(qParam);
			// Clear URL params after message is sent
			clearUrlParams();
		} else if (modelParam || newChatParam === 'true') {
			// Clear params even if no message was sent (just model selection or new_chat)
			clearUrlParams();
		}
	}

	onMount(async () => {
		if (!isConversationsInitialized()) {
			await conversationsStore.initialize();
		}

		clearActiveConversation();
		clearUIState();

		// Handle URL params only if we have ?q= or ?model= or ?new_chat=true
		if (qParam !== null || modelParam !== null || newChatParam === 'true') {
			await handleUrlParams();
		}
	});
</script>

<svelte:head>
	<title>llama.cpp - AI Chat Interface</title>
</svelte:head>

<ChatScreen showCenteredEmpty={true} />

<DialogModelNotAvailable
	bind:open={showModelNotAvailable}
	modelName={requestedModelName}
	availableModels={availableModelNames}
/>
