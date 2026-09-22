# Windows setup for the SmartRouter branch

This branch is published at `https://github.com/sokolovxf/free-claude-code` on
`smart-router-v2`. The commands below update an existing Windows installation
without copying anyone's API keys.

## Update FCC to this branch

Open PowerShell and stop the running FCC desktop/server process first:

```powershell
$installScript = "https://raw.githubusercontent.com/sokolovxf/free-claude-code/smart-router-v2/scripts/install.ps1"
& ([scriptblock]::Create((irm $installScript)))
```

The installer keeps the existing `%USERPROFILE%\.fcc\.env` credentials and
installs the SmartRouter-enabled package from this branch.

## Apply the same route pool

Clone the branch so the profile merge script is available:

```powershell
$checkout = Join-Path $HOME "free-claude-code-smart-router"
if (Test-Path $checkout) {
    git -C $checkout fetch origin smart-router-v2
    git -C $checkout switch smart-router-v2
    git -C $checkout pull --ff-only origin smart-router-v2
} else {
    git clone --branch smart-router-v2 https://github.com/sokolovxf/free-claude-code.git $checkout
}

& (Join-Path $checkout "scripts\apply-smart-router-profile.ps1") `
    -ProfilePath (Join-Path $checkout ".env.smart-router.example")
```

The script backs up the existing config, replaces only the router profile
(`MODEL`, `MODEL_FALLBACKS`, `PORT`, and the verified-free list), and preserves
all existing provider credentials. If a provider key is missing, add the
friend's own key through the Admin UI; never copy your `.env` file or API keys.

## Start and verify

Restart FCC after applying the profile:

```powershell
fcc-server
```

Then, in another PowerShell window:

```powershell
fcc-claude
```

Open the Admin UI and confirm the configured port is `8083`. Use `fcc-health`
to inspect route health, quota state, fallback order, and the currently
selected model.

The SmartRouter code is shared by the package; provider availability and quota
state are account-specific, so your friend must use his own provider accounts.
