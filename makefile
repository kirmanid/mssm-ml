mlp: mlp.cpp
	g++ mlp.cpp -o mlp $(pkg-config --cflags eigen3)
clean: 
	$(RM) mlp

