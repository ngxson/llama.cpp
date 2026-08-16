<script lang="ts">
	import { AlertCircle, Loader2, Wifi, WifiOff } from '@lucide/svelte';
	import { SettingsGroup } from '$lib/components/app/settings';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { webrtcStore } from '$lib/stores/webrtc.svelte';
	import { fade } from 'svelte/transition';

	let joinInput = $state('');
	let joinError = $state('');
	let joining = $state(false);

	async function handleJoin() {
		joinError = '';
		const code = joinInput.trim().replace(/\s/g, '');

		if (code.length < 40) {
			joinError = 'Code must be 40 characters';

			return;
		}

		joining = true;
		try {
			await webrtcStore.joinAsClient(code);
		} catch (e) {
			joinError = e instanceof Error ? e.message : String(e);
		} finally {
			joining = false;
		}
	}

	function handleLeave() {
		webrtcStore.leaveAsClient();
		joinInput = '';
		joinError = '';
	}
</script>

<div class="space-y-12" in:fade={{ duration: 150 }}>
	<SettingsGroup title="Join">
		<div class="space-y-4">
			<p class="text-sm text-muted-foreground">
				Connect to a remote llama.cpp instance running llama-connect. All requests will be routed
				through the peer-to-peer tunnel.
			</p>

			{#if webrtcStore.mode === 'client'}
				<div class="space-y-3">
					<div class="flex items-center gap-3">
						{#if webrtcStore.status === 'connecting'}
							<Badge variant="secondary" class="gap-1.5">
								<Loader2 class="h-3 w-3 animate-spin" />
								Connecting...
							</Badge>
						{:else if webrtcStore.status === 'connected'}
							<Badge variant="default" class="gap-1.5">
								<Wifi class="h-3 w-3" />
								Connected to host
							</Badge>
						{:else if webrtcStore.status === 'error'}
							<Badge variant="destructive" class="gap-1.5">
								<AlertCircle class="h-3 w-3" />
								Disconnected
							</Badge>
							<span class="text-sm text-destructive">{webrtcStore.errorMessage}</span>
						{/if}
					</div>

					<Button variant="outline" onclick={handleLeave}>
						<WifiOff class="h-4 w-4" />
						Leave
					</Button>
				</div>
			{:else}
				<div class="space-y-3">
					<div class="space-y-1.5">
						<label for="join-code" class="text-sm font-medium">Access code</label>
						<Input
							id="join-code"
							placeholder="Paste the 40-character code from the host"
							bind:value={joinInput}
							disabled={joining}
							class="font-mono"
						/>
						{#if joinError}
							<p class="text-sm text-destructive">{joinError}</p>
						{/if}
					</div>

					<Button onclick={handleJoin} disabled={joining || joinInput.trim().length < 40}>
						{#if joining}
							<Loader2 class="h-4 w-4 animate-spin" />
							Connecting...
						{:else}
							<Wifi class="h-4 w-4" />
							Connect
						{/if}
					</Button>
				</div>
			{/if}
		</div>
	</SettingsGroup>

	<p class="text-xs text-muted-foreground">
		Uses WebRTC with Google STUN servers for NAT traversal. Signaling via public WebTorrent
		trackers. No data is routed through any relay server.
	</p>
</div>
