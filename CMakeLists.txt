set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 2.8)
project( feature_matcher)

find_package(OpenMP)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
find_package( OpenCV REQUIRED )

#add_library( YourLib ${OpenCV_LIBS})

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(feature_matcher_sift_parallel feature_matcher_sift_parallel.cpp )
target_link_libraries( feature_matcher_sift_parallel ${OpenCV_LIBS} )

add_executable(feature_matcher_sift_mpi feature_matcher_sift_mpi.cpp )
target_link_libraries( feature_matcher_sift_mpi ${OpenCV_LIBS} ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
	set_target_properties(feature_matcher_sift_mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
	set_target_properties(feature_matcher_sift_mpi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()



