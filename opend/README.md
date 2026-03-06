# FutuOpenD Gateway Configuration

FutuOpenD is the required gateway to communicate with Futu OpenAPI.

## Installation

1. Download the correct version of FutuOpenD for your OS (macOS) from the official website:
   https://www.futunn.com/download/openAPI

2. Extract and place the `FutuOpenD` binary in this directory (`opend/`).

3. First-time login:
   Run the binary interactively to enter your account credentials and verification code.
   ```bash
   ./FutuOpenD -login_account=YOUR_PHONE -login_pwd_md5=YOUR_MD5_PWD
   ```
   *Note: For macOS, use the `.dmg` or `.pkg` and grab the binary, or use the UI version and enable API listening.*

## Automatic Login (Production)

Once you have successfully logged in, FutuOpenD saves a token in `FutuOpenD.xml`.
You can then run it in the background:

```bash
nohup ./FutuOpenD > opend.log 2>&1 &
```

## Docker Warning
Because the initial login requires SMS/Device verification, it is highly recommended to run FutuOpenD natively on your macOS host machine rather than inside Docker. The Python container will connect to it via `host.docker.internal:11111`.
