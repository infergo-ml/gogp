all: selfcheck
	
GO=go

STUDIES=barebones hyperpriors warpedtime events anynoise

selfcheck:
	for study in $(STUDIES); do \
		(cd $$study && make GO=$(GO)) \
	done

clean:
	for study in $(STUDIES); do \
		(cd $$study && make clean) \
	done

