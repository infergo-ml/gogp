all: build

build: kernel/ad/kernel.go
	go build ./...

test: kernel/ad/kernel.go
	go test ./gp ./kernel

kernel/ad/kernel.go: kernel/kernel.go kernel/noise.go
	deriv kernel

clean:
	rm -f kernel/ad/*.go

push:
	for repo in origin ssh://git@github.com/infergo-ml/gogp; do git push $$repo; git push --tags $$repo; done

