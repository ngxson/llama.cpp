<script lang="ts">
	import { ChatScreen } from '$lib/components/app';
	import { sendMessage, clearUIState } from '$lib/stores/chat.svelte';
	import {
		conversationsStore,
		isConversationsInitialized,
		clearActiveConversation,
		createConversation
	} from '$lib/stores/conversations.svelte';
	import { onMount } from 'svelte';
	import { page } from '$app/state';

	let qParam = $derived(page.url.searchParams.get('q'));

	onMount(async () => {
		if (!isConversationsInitialized()) {
			await conversationsStore.initialize();
		}

		clearActiveConversation();
		clearUIState();

		if (qParam !== null) {
			await createConversation();
			await sendMessage(qParam);
		}
	});
</script>

<svelte:head>
	<title>llama.cpp - AI Chat Interface</title>
</svelte:head>

<ChatScreen showCenteredEmpty={true} />
