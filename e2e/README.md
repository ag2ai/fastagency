# Playwright tests

When running from dev container, VSCode should have Playwright extension installed,
and tests can be run directly from it, under testing icon on the far left.

Currently in dev container tests are run in
headless mode, so do not try ```Show browser``` option in VS Code, unless you wish
to restart machine anyways.

Currently there are 3 Playwright configurations available, they differ in which server fastagency
server is used, and where are corresponding playwrigth tests located:
- llm-sans (no llms in play just mesop ui testing)
- llm (end to end testing with llms in play)
- gunicorn (run fastagency with gunicorn)

When switching playwright configuration in VS Code, execute ```Run global teardown``` and ```Run global setup```,
since they do not seem to be executed by configuration switch.

Directory ```e2e/llm-sans``` contain tests that do not involve llms, and typically run faster,
but concentrate on testing Mesop functionality only.

Directory ```e2e/llm``` contain tests (actually currently only one) that do involve llms, and test
integration of Mesop with Providers running Autogen


To run non llm tests select playwright.llm-sans.config.ts in VS Code,
or in shell run:
```
npx playwright test -c "playwright.llm-sans.config.ts"
```

To run tests that require llms select playwright.llm.config.ts in VS Code,
or in shell run:
```
npx playwright test -c "playwright.llm.config.ts"
```
Note that llms tests take longer to execute, and sometimes they do timeout.

To run non llm tests with gunicorn select playwright.gunicorn.config.ts in VS Code,
or in shell run:
```
npx playwright test -c "playwright.gunicorn.config.ts"
```
note that gunicorn for some reason takes some time to spin-up, and this can also cause
tests to run slower, and sometimes even to error due to time out.

## Github test workflow integration

* Currently llm-sans set of test is run inside test workflow.
* Unless for Python 3.10 which is not supported
*  Code coverage data is not collected
