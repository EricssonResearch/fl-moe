DOCKER_IMAGE_VERSION=latest
DOCKER_IMAGE_NAME=martisak/fl-moe
DOCKER_IMAGE_TAGNAME=$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION)
ENVIRONMENT := $(dir $(abspath $(dir $$PWD)))

default: build

pull:
	docker pull nvidia/cuda:10.2-base-ubuntu18.04

push:
	docker push $(DOCKER_IMAGE_TAGNAME)

build: pull
	docker build -t $(DOCKER_IMAGE_TAGNAME) .
	# docker tag $(DOCKER_IMAGE_TAGNAME) $(DOCKER_IMAGE_NAME):latest

test:
	docker run --rm $(DOCKER_IMAGE_TAGNAME) /bin/echo "Success."

rmi:
	docker rmi -f $(DOCKER_IMAGE_TAGNAME)

debug:
	docker run --rm -it -v $(ENVIRONMENT):/home/user/src $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION) bash

rebuild: rmi build