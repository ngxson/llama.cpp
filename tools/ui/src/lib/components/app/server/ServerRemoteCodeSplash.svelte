<script lang="ts">
	import { Loader2, Radio, Wifi } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { APP_NAME, ICON_CLASS_DEFAULT } from '$lib/constants';
	import { KeyboardKey } from '$lib/enums';
	import { webrtcStore } from '$lib/stores/webrtc.svelte';
	import { fade } from 'svelte/transition';

	let code = $state('');
	let error = $state('');
	let connecting = $state(false);

	async function handleConnect() {
		error = '';

		const shareCode = code.trim().replace(/\s/g, '');

		if (shareCode.length < 40) {
			error = 'Code must be 40 characters';

			return;
		}

		connecting = true;
		try {
			await webrtcStore.joinAsClient(shareCode);
			// The app read nothing from a server yet, so start it over with the
			// tunnel already in place.
			location.reload();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			connecting = false;
		}
	}

	function handleUseAnotherCode() {
		webrtcStore.leaveAsClient();
		code = '';
		error = '';
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ENTER) {
			handleConnect();
		}
	}
</script>

<svelte:head>
	<title>Connect - {APP_NAME}</title>
</svelte:head>

<div class="flex h-dvh items-center justify-center">
	<div class="w-full max-w-md px-4" in:fade={{ duration: 300 }}>
		<div class="mb-6 text-center">
			<div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
				<Radio class="h-8 w-8" />
			</div>

			<h1 class="mb-2 text-xl font-semibold">Connect to your llama.cpp server</h1>

			<p class="text-sm text-muted-foreground">
				This build has no server of its own. Run <span class="font-mono">llama-connect</span> next to
				your llama-server and paste the access code it prints.
			</p>
		</div>

		<div class="space-y-3">
			{#if webrtcStore.status === 'connecting'}
				<div class="flex items-center justify-center gap-2 py-2 text-sm text-muted-foreground">
					<Loader2 class="{ICON_CLASS_DEFAULT} animate-spin" />
					Connecting to your server...
				</div>

				<Button variant="outline" onclick={handleUseAnotherCode} class="w-full">
					Use another code
				</Button>
			{:else}
				<div class="space-y-1.5">
					<Label for="connect-code" class="text-sm font-medium">Access code</Label>

					<Input
						id="connect-code"
						placeholder="Paste the 40-character code"
						bind:value={code}
						onkeydown={handleKeydown}
						disabled={connecting}
						class="font-mono"
					/>

					{#if error}
						<p class="text-sm text-destructive">{error}</p>
					{:else if webrtcStore.status === 'error' && webrtcStore.errorMessage}
						<p class="text-sm text-destructive">{webrtcStore.errorMessage}</p>
					{/if}
				</div>

				<Button
					onclick={handleConnect}
					disabled={connecting || code.trim().length < 40}
					class="w-full"
				>
					{#if connecting}
						<Loader2 class="{ICON_CLASS_DEFAULT} animate-spin" />
						Connecting...
					{:else}
						<Wifi class={ICON_CLASS_DEFAULT} />
						Connect
					{/if}
				</Button>
			{/if}
		</div>

		<p class="mt-6 text-center text-xs text-muted-foreground">
			Uses WebRTC with Google STUN servers for NAT traversal. Signaling via public WebTorrent
			trackers. No data is routed through any relay server.
		</p>
	</div>
</div>
