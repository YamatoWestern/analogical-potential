"""
Constant values in the project
"""

positive_permute = [
    [0,1,2,3], [1,0,3,2], [2,0,3,1], [3,2,1,0],
    [0,2,1,3], [1,3,0,2], [2,3,0,1], [3,1,2,0], 
]

negative_permute = [
    [0,1,3,2], [1,0,2,3], [2,1,3,0], [3,0,2,1],
    [0,3,1,2], [1,2,0,3], [2,3,1,0], [3,2,0,1],
]

reverse_permute = [
    [0,2,3,1], [1,2,3,0], [2,0,1,3], [3,0,1,2],
    [0,3,2,1], [1,3,2,0], [2,1,0,3], [3,1,0,2],
]