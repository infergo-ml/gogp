all: selfcheck

STUDIES=barebones hyperpriors warpedtime events anynoise

selfcheck:
	for study in $(STUDIES); do \
		(cd $$study && make) \
	done

clean:
	for study in $(STUDIES); do \
		(cd $$study && make clean) \
	done

