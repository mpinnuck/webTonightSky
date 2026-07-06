# Available targets:
#   make zip-source  - Create TonightSky_source.zip using zip_source.sh.
#   make deploy      - Run delpoy to deploy changed files to production.
#   make deploy-dry  - Run delpoy in dry-run mode.

.PHONY: zip-source deploy deploy-dry

ZIP_SH    := ./zip_source.sh
DEPLOY_SH := ./deploy.sh

zip-source:
	@$(ZIP_SH)

deploy:
	@$(DEPLOY_SH)

deploy-dry:
	@$(DEPLOY_SH) --dry-run