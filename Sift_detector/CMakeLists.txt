set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 2.8)
project( sift_detector )

find_package(OpenMP)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
find_package( OpenCV REQUIRED )


if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


add_executable( sift_detector sift_detector.cpp )
target_link_libraries( sift_detector ${OpenCV_LIBS})

add_executable( sift_detector_script sift_detector_script.cpp )
target_link_libraries( sift_detector_script ${OpenCV_LIBS})

add_executable( sift_detector_parallel_mpi sift_detector_parallel_mpi.cpp )
target_link_libraries( sift_detector_parallel_mpi ${OpenCV_LIBS} ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
	set_target_properties(sift_detector_parallel_mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
	set_target_properties(sift_detector_parallel_mpi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
