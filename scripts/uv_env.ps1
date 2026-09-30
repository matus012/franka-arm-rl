# uv locations outside %APPDATA% (the Claude desktop app virtualizes AppData writes).
# Same values as ws\robotics_base\scripts\uv_env.ps1. Dot-source before any uv call.
$env:UV_PYTHON_INSTALL_DIR = "$env:USERPROFILE\.uv\python"
$env:UV_TOOL_DIR = "$env:USERPROFILE\.uv\tools"
$env:UV_CACHE_DIR = "$env:USERPROFILE\.uv\cache"
if (($env:PATH -split ';') -notcontains "$env:USERPROFILE\.local\bin") { $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH" }
