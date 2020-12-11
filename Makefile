all:
	$(MAKE) -C nets/ROIPooling all

clean:
	$(MAKE) -C nets/ROIPooling clean
	rm -rf `find -iname __pycache__`