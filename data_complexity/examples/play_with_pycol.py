from sklearn.datasets import make_classification, make_moons
from data_complexity.pycol import Complexity

# X, y = make_classification(
#     n_samples=200,
#     n_features=5,
#     n_informative=3,
#     n_redundant=0,
#     n_classes=2,
#     random_state=42
# )
X, y = make_moons(noise=0.1, random_state=0)

dataset = {'X': X, 'y': y}
complexity = Complexity(dataset=dataset, file_type="array")
# complexity = Complexity("dataset/61_iris.arff",distance_func="default",file_type="arff")
print('Dataset loaded\n')

# Feature Overlap
print('\nFeature Overlap Measures (metric per feature):')
print(f'F1: {complexity.F1()}')
print(f'F1v: {complexity.F1v()}')
print(f'F2: {complexity.F2()}')
print(f'F3: {complexity.F3()}')
print(f'F4: {complexity.F4()}')
# (...)

# Instance Overlap
print('\nInstance Overlap Measures:')
print(f'Raug: {complexity.R_value()}')
print(f'deg_overlap: {complexity.deg_overlap()}')
print(f'N3: {complexity.N3()}')
print(f'SI: {complexity.SI()}')
print(f'N4: {complexity.N4()}')
print(f'kDN: {complexity.kDN()}')
print(f'D3: {complexity.D3_value()}')
print(f'CM: {complexity.CM()}')
# (...)

# Structural Overlap
print('\nStructural Overlap Measures:')
print(f'N1: {complexity.N1()}')
print(f'T1: {complexity.T1()}')
print(f'Clust: {complexity.Clust()}')
# (...)

# Multiresolution Overlap
print('\nMultiresolution Overlap Measures:')
print(f'MRCA: {complexity.MRCA()}')
print(f'C1: {complexity.C1()}')
print(f'Purity: {complexity.purity()}')
# (...)


feature_overlap, f_names = complexity.feature_overlap(viz=False)
instance_overlap, i_names = complexity.instance_overlap(viz=False)
structural_overlap, s_names = complexity.structure_overlap(viz=False)

print('\nFeature Overlap Measures:')
for measure, value in zip(f_names, feature_overlap):
    print(f'    - {measure}: {value}')

print('\nInstance Overlap Measures:')
for measure, value in zip(i_names, instance_overlap):
    print(f'    - {measure}: {value}')

print('\nStructural Overlap Measures:')
for measure, value in zip(s_names, structural_overlap):
    print(f'    - {measure}: {value}')