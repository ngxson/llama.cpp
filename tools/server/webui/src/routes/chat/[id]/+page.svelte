<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { afterNavigate } from '$app/navigation';
	import { ChatScreen } from '$lib/components/app';
	import { isLoading, stopGeneration, syncLoadingStateForChat } from '$lib/stores/chat.svelte';
	import {
		activeConversation,
		activeMessages,
		loadConversation
	} from '$lib/stores/conversations.svelte';
	import { selectModel, modelOptions, selectedModelId } from '$lib/stores/models.svelte';

	let chatId = $derived(page.params.id);
	let currentChatId: string | undefined = undefined;

	async function selectModelFromLastAssistantResponse() {
		const messages = activeMessages();
		if (messages.length === 0) return;

		let lastMessageWithModel: DatabaseMessage | undefined;

		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].model) {
				lastMessageWithModel = messages[i];
				break;
			}
		}

		if (!lastMessageWithModel) return;

		const currentModelId = selectedModelId();
		const currentModelName = modelOptions().find((m) => m.id === currentModelId)?.model;

		if (currentModelName === lastMessageWithModel.model) {
			return;
		}

		const matchingModel = modelOptions().find(
			(option) => option.model === lastMessageWithModel.model
		);

		if (matchingModel) {
			try {
				await selectModel(matchingModel.id);
				console.log(`Automatically loaded model: ${lastMessageWithModel.model} from last message`);
			} catch (error) {
				console.warn('Failed to automatically select model from last message:', error);
			}
		}
	}

	afterNavigate(() => {
		setTimeout(() => {
			selectModelFromLastAssistantResponse();
		}, 100);
	});

	$effect(() => {
		if (chatId && chatId !== currentChatId) {
			currentChatId = chatId;

			// Skip loading if this conversation is already active (e.g., just created)
			if (activeConversation()?.id === chatId) {
				return;
			}

			(async () => {
				const success = await loadConversation(chatId);
				if (success) {
					syncLoadingStateForChat(chatId);
				} else {
					await goto('#/');
				}
			})();
		}
	});

	$effect(() => {
		if (typeof window !== 'undefined') {
			const handleBeforeUnload = () => {
				if (isLoading()) {
					console.log('Page unload detected while streaming - aborting stream');
					stopGeneration();
				}
			};

			window.addEventListener('beforeunload', handleBeforeUnload);

			return () => {
				window.removeEventListener('beforeunload', handleBeforeUnload);
			};
		}
	});
</script>

<svelte:head>
	<title>{activeConversation()?.name || 'Chat'} - llama.cpp</title>
</svelte:head>

<ChatScreen />
