all: selfcheck

GO=go

selfcheck: anynoise
	./anynoise selfcheck

anynoise: kernel/ad/kernel.go model/ad/model.go main.go ../tutorial.go
	$(GO) build .

kernel/ad/kernel.go: kernel/kernel.go
	deriv kernel

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./anynoise {kernel,model}/ad/*.go 
