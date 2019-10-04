all: selfcheck

selfcheck: events
	./events -events 4:4.5:0.1,6:7:0.5 selfcheck

events: kernel/ad/kernel.go main.go ../tutorial.go
	go build .

kernel/ad/kernel.go: kernel/kernel.go
	deriv kernel

clean:
	rm -f ./events kernel/ad/*.go
