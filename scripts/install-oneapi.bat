::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT
::
::  Vendored from llama.cpp (scripts/install-oneapi.bat) — Intel's official
::  silent-install helper for the oneAPI offline bootstrapper. Takes:
::    %1 = URL of the offline installer .exe
::    %2 = colon-separated component list (may be empty for full install)
::  Used by .github/workflows/capi-build.yml for the Windows SYCL job.

set URL=%1
set COMPONENTS=%2

:: Fail fast instead of hanging forever: --fail turns HTTP 4xx/5xx into a
:: non-zero exit (a 403 page otherwise lands in webimage.exe with rc 0);
:: --speed-limit/--speed-time abort a stalled mid-transfer connection (the
:: 1.1 GB Essentials image has hung silently for hours on Intel's CDN);
:: --max-time caps the whole download. Kept loose enough for a slow runner.
echo downloading oneAPI webimage from %URL%
curl.exe --fail --location --continue-at - --output %TEMP%\webimage.exe --url %URL% --retry 5 --retry-delay 5 --connect-timeout 30 --speed-limit 10240 --speed-time 60 --max-time 1800
if %ERRORLEVEL% neq 0 (
  echo oneAPI webimage download failed, curl exit code %ERRORLEVEL%
  exit /b %ERRORLEVEL%
)
for %%F in (%TEMP%\webimage.exe) do @echo download ok: %%~zF bytes
start /b /wait %TEMP%\webimage.exe -s -x -f webimage_extracted --log extract.log
if %ERRORLEVEL% neq 0 (
  echo oneAPI webimage extraction failed, exit code %ERRORLEVEL%
  exit /b %ERRORLEVEL%
)
del %TEMP%\webimage.exe
if "%COMPONENTS%"=="" (
  webimage_extracted\bootstrapper.exe -s --action install --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 -p=NEED_VS2022_INTEGRATION=0 --log-dir=.
) else (
  webimage_extracted\bootstrapper.exe -s --action install --components=%COMPONENTS% --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 -p=NEED_VS2022_INTEGRATION=0 --log-dir=.
)
set installer_exit_code=%ERRORLEVEL%
rd /s/q "webimage_extracted"
exit /b %installer_exit_code%
