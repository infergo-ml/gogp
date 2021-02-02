all: test build

GO=go

build: kernel/ad/kernel.go
	$(GO) build ./gp ./kernel ./tutorial

test: kernel/ad/kernel.go
	$(GO) test ./gp ./kernel ./tutorial

kernel/ad/kernel.go: kernel/kernel.go kernel/noise.go
	deriv kernel

clean:
	rm -f kernel/ad/*.go

push:
	for repo in origin ssh://git@github.com/infergo-ml/gogp; do git push $$repo; git push --tags $$repo; done

