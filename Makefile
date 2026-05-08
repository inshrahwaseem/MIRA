.PHONY: setup dev test lint deploy db-migrate

setup:
	pnpm install
	cd services/ml-service && pip install -r requirements.txt
	cd services/llm-service && pip install -r requirements.txt

dev:
	docker-compose up --build

test:
	pnpm run test
	cd services/ml-service && pytest
	cd services/llm-service && pytest

lint:
	pnpm run lint
	cd services/ml-service && ruff check . && mypy .
	cd services/llm-service && ruff check . && mypy .

deploy:
	vercel deploy --prod
	railway up

db-migrate:
	pnpm --filter api-gateway run migrate
