??
?-?-
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
?
AsString

input"T

output"
Ttype:
2		
"
	precisionint?????????"

scientificbool( "
shortestbool( "
widthint?????????"
fillstring 
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
?
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 ?
?
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0?????????"
value_indexint(0?????????"+

vocab_sizeint?????????(0?????????"
	delimiterstring	?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
2
LookupTableSizeV2
table_handle
size	?
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
8
Minimum
x"T
y"T
z"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
?
SparseSegmentSum	
data"T
indices"Tidx
segment_ids
output"T"
Ttype:
2	"
Tidxtype0:
2	
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
c
StringSplit	
input
	delimiter
indices	

values	
shape	"

skip_emptybool(
:
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?
9
VarIsInitializedOp
resource
is_initialized
?
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.1.02unknown8??

global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 
?
global_stepVarHandleOp*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: *
shared_nameglobal_step
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 
_
global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
dtype0	
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0	
o
input_example_tensorPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB 
d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB 
j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB 
z
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0* 
valueBBdense_input
j
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB 
?
ParseExample/ParseExampleV2ParseExampleV2input_example_tensor!ParseExample/ParseExampleV2/names'ParseExample/ParseExampleV2/sparse_keys&ParseExample/ParseExampleV2/dense_keys'ParseExample/ParseExampleV2/ragged_keysParseExample/Const*
Tdense
2*#
_output_shapes
:?????????*
dense_shapes
: *

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 
?
ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAtransform_fn\assets\vocab_compute_and_apply_vocabulary_vocabulary
S
transform/ConstConst*
_output_shapes
: *
dtype0	*
valueB		 R??
?
transform/Const_1Const*
_output_shapes
: *
dtype0*z
valueqBo BiC:\Users\the\AppData\Local\Temp\tmp7ss2kt33\tftransform_tmp\vocab_compute_and_apply_vocabulary_vocabulary
?
(transform/transform/inputs/F_dense_inputPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
{
 transform/transform/inputs/labelPlaceholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
4transform/transform/inputs/inputs/F_dense_input_copyIdentityParseExample/ParseExampleV2*
T0*#
_output_shapes
:?????????
?
,transform/transform/inputs/inputs/label_copyIdentity transform/transform/inputs/label*
T0	*#
_output_shapes
:?????????
m
%transform/transform/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B.,!?() 
?
+transform/transform/StringSplit/StringSplitStringSplit4transform/transform/inputs/inputs/F_dense_input_copy%transform/transform/StringSplit/Const*<
_output_shapes*
(:?????????:?????????:
?
Itransform/transform/compute_and_apply_vocabulary/vocabulary/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Ctransform/transform/compute_and_apply_vocabulary/vocabulary/ReshapeReshape-transform/transform/StringSplit/StringSplit:1Itransform/transform/compute_and_apply_vocabulary/vocabulary/Reshape/shape*
T0*#
_output_shapes
:?????????
?
}transform/transform/compute_and_apply_vocabulary/vocabulary/vocab_compute_and_apply_vocabulary_vocabulary_unpruned_vocab_sizePlaceholder*
_output_shapes
: *
dtype0	*
shape: 
?
Gtransform/transform/compute_and_apply_vocabulary/vocabulary/PlaceholderPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
Btransform/transform/compute_and_apply_vocabulary/apply_vocab/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
Gtransform/transform/compute_and_apply_vocabulary/apply_vocab/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*y
shared_namejhhash_table_Tensor("compute_and_apply_vocabulary/vocabulary/Placeholder:0", shape=(), dtype=string)_-2_-1*
value_dtype0	
?
itransform/transform/compute_and_apply_vocabulary/apply_vocab/text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2Gtransform/transform/compute_and_apply_vocabulary/apply_vocab/hash_tableConst*
	key_index?????????*
value_index?????????
?
^transform/transform/compute_and_apply_vocabulary/apply_vocab/hash_table_Size/LookupTableSizeV2LookupTableSizeV2Gtransform/transform/compute_and_apply_vocabulary/apply_vocab/hash_table*
_output_shapes
: 
?
`transform/transform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Gtransform/transform/compute_and_apply_vocabulary/apply_vocab/hash_table-transform/transform/StringSplit/StringSplit:1Btransform/transform/compute_and_apply_vocabulary/apply_vocab/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
Dtransform/transform/compute_and_apply_vocabulary/apply_vocab/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Btransform/transform/compute_and_apply_vocabulary/apply_vocab/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
?
@transform/transform/compute_and_apply_vocabulary/apply_vocab/subSub^transform/transform/compute_and_apply_vocabulary/apply_vocab/hash_table_Size/LookupTableSizeV2Btransform/transform/compute_and_apply_vocabulary/apply_vocab/sub/y*
T0	*
_output_shapes
: 
?
Ftransform/transform/compute_and_apply_vocabulary/apply_vocab/Minimum/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
Dtransform/transform/compute_and_apply_vocabulary/apply_vocab/MinimumMinimumDtransform/transform/compute_and_apply_vocabulary/apply_vocab/Const_1Ftransform/transform/compute_and_apply_vocabulary/apply_vocab/Minimum/y*
T0	*
_output_shapes
: 
?
Ftransform/transform/compute_and_apply_vocabulary/apply_vocab/Maximum/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
Dtransform/transform/compute_and_apply_vocabulary/apply_vocab/MaximumMaximum@transform/transform/compute_and_apply_vocabulary/apply_vocab/subFtransform/transform/compute_and_apply_vocabulary/apply_vocab/Maximum/y*
T0	*
_output_shapes
: 
 
transform/transform/initNoOp
"
transform/transform/init_1NoOp

transform/initNoOp
?
Ilinear/linear_model/dense_input/weights/Initializer/zeros/shape_as_tensorConst*:
_class0
.,loc:@linear/linear_model/dense_input/weights*
_output_shapes
:*
dtype0*
valueB"!N     
?
?linear/linear_model/dense_input/weights/Initializer/zeros/ConstConst*:
_class0
.,loc:@linear/linear_model/dense_input/weights*
_output_shapes
: *
dtype0*
valueB
 *    
?
9linear/linear_model/dense_input/weights/Initializer/zerosFillIlinear/linear_model/dense_input/weights/Initializer/zeros/shape_as_tensor?linear/linear_model/dense_input/weights/Initializer/zeros/Const*
T0*:
_class0
.,loc:@linear/linear_model/dense_input/weights* 
_output_shapes
:
??
?
'linear/linear_model/dense_input/weightsVarHandleOp*:
_class0
.,loc:@linear/linear_model/dense_input/weights*
_output_shapes
: *
dtype0*
shape:
??*8
shared_name)'linear/linear_model/dense_input/weights
?
Hlinear/linear_model/dense_input/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/dense_input/weights*
_output_shapes
: 
?
.linear/linear_model/dense_input/weights/AssignAssignVariableOp'linear/linear_model/dense_input/weights9linear/linear_model/dense_input/weights/Initializer/zeros*
dtype0
?
;linear/linear_model/dense_input/weights/Read/ReadVariableOpReadVariableOp'linear/linear_model/dense_input/weights* 
_output_shapes
:
??*
dtype0
?
2linear/linear_model/bias_weights/Initializer/zerosConst*3
_class)
'%loc:@linear/linear_model/bias_weights*
_output_shapes
:*
dtype0*
valueB*    
?
 linear/linear_model/bias_weightsVarHandleOp*3
_class)
'%loc:@linear/linear_model/bias_weights*
_output_shapes
: *
dtype0*
shape:*1
shared_name" linear/linear_model/bias_weights
?
Alinear/linear_model/bias_weights/IsInitialized/VarIsInitializedOpVarIsInitializedOp linear/linear_model/bias_weights*
_output_shapes
: 
?
'linear/linear_model/bias_weights/AssignAssignVariableOp linear/linear_model/bias_weights2linear/linear_model/bias_weights/Initializer/zeros*
dtype0
?
4linear/linear_model/bias_weights/Read/ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
_output_shapes
:*
dtype0
?
Rlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/Shape/CastCast-transform/transform/StringSplit/StringSplit:2*

DstT0*

SrcT0	*
_output_shapes
:
?
[linear/linear_model/linear/linear_model/linear/linear_model/dense_input/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
]linear/linear_model/linear/linear_model/linear/linear_model/dense_input/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
]linear/linear_model/linear/linear_model/linear/linear_model/dense_input/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Ulinear/linear_model/linear/linear_model/linear/linear_model/dense_input/strided_sliceStridedSliceRlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/Shape/Cast[linear/linear_model/linear/linear_model/linear/linear_model/dense_input/strided_slice/stack]linear/linear_model/linear/linear_model/linear/linear_model/dense_input/strided_slice/stack_1]linear/linear_model/linear/linear_model/linear/linear_model/dense_input/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Plinear/linear_model/linear/linear_model/linear/linear_model/dense_input/Cast/x/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
Nlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/Cast/xPackUlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/strided_slicePlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/Cast/x/1*
N*
T0*
_output_shapes
:
?
Llinear/linear_model/linear/linear_model/linear/linear_model/dense_input/CastCastNlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/Cast/x*

DstT0	*

SrcT0*
_output_shapes
:
?
Ulinear/linear_model/linear/linear_model/linear/linear_model/dense_input/SparseReshapeSparseReshape+transform/transform/StringSplit/StringSplit-transform/transform/StringSplit/StringSplit:2Llinear/linear_model/linear/linear_model/linear/linear_model/dense_input/Cast*-
_output_shapes
:?????????:
?
^linear/linear_model/linear/linear_model/linear/linear_model/dense_input/SparseReshape/IdentityIdentity`transform/transform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/LookupTableFindV2*
T0	*#
_output_shapes
:?????????
?
`linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
_linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Zlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SliceSliceWlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/SparseReshape:1`linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice/begin_linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Zlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ylinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/ProdProdZlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SliceZlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Const*
T0	*
_output_shapes
: 
?
elinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
blinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
]linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2GatherV2Wlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/SparseReshape:1elinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2/indicesblinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
[linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Cast/xPackYlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Prod]linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2*
N*
T0	*
_output_shapes
:
?
blinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseReshapeSparseReshapeUlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/SparseReshapeWlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/SparseReshape:1[linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
klinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseReshape/IdentityIdentity^linear/linear_model/linear/linear_model/linear/linear_model/dense_input/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
clinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
alinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GreaterEqualGreaterEqualklinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseReshape/Identityclinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Zlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/WhereWherealinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
blinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
\linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/ReshapeReshapeZlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Whereblinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
dlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
_linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2_1GatherV2blinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseReshape\linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Reshapedlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
dlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
_linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2_2GatherV2klinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseReshape/Identity\linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Reshapedlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
]linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/IdentityIdentitydlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
nlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
|linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows_linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2_1_linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/GatherV2_2]linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Identitynlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
?linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
?linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
zlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlice|linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows?linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/strided_slice/stack?linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1?linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
qlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/CastCastzlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/strided_slice*

DstT0*

SrcT0	*#
_output_shapes
:?????????
?
slinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/UniqueUnique~linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
}linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather'linear/linear_model/dense_input/weightsslinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*:
_class0
.,loc:@linear/linear_model/dense_input/weights*'
_output_shapes
:?????????*
dtype0
?
?linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentity}linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*:
_class0
.,loc:@linear/linear_model/dense_input/weights*'
_output_shapes
:?????????
?
?linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
llinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparseSparseSegmentSum?linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1ulinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/Unique:1qlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:?????????
?
dlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
^linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Reshape_1Reshape~linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2dlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Zlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/ShapeShapellinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
hlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
jlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
jlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
blinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/strided_sliceStridedSliceZlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Shapehlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/strided_slice/stackjlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/strided_slice/stack_1jlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
\linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
Zlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/stackPack\linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/stack/0blinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/strided_slice*
N*
T0*
_output_shapes
:
?
Ylinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/TileTile^linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Reshape_1Zlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
_linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/zeros_like	ZerosLikellinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Tlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sumSelectYlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Tile_linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/zeros_likellinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
[linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Cast_1CastWlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
?
blinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
alinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
\linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_1Slice[linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Cast_1blinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_1/beginalinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
\linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Shape_1ShapeTlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum*
T0*
_output_shapes
:
?
blinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
alinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
\linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_2Slice\linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Shape_1blinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_2/beginalinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
`linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
[linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/concatConcatV2\linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_1\linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Slice_2`linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/concat/axis*
N*
T0*
_output_shapes
:
?
^linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Reshape_2ReshapeTlinear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum[linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Plinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum_no_biasIdentity^linear/linear_model/linear/linear_model/linear/linear_model/dense_input/weighted_sum/Reshape_2*
T0*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum/ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
_output_shapes
:*
dtype0
?
Hlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sumBiasAddPlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum_no_biasWlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum/ReadVariableOp*
T0*'
_output_shapes
:?????????
k
ReadVariableOpReadVariableOp linear/linear_model/bias_weights*
_output_shapes
:*
dtype0
]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
strided_sliceStridedSliceReadVariableOpstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
N
	bias/tagsConst*
_output_shapes
: *
dtype0*
valueB
 Bbias
P
biasScalarSummary	bias/tagsstrided_slice*
T0*
_output_shapes
: 
?
,zero_fraction/total_size/Size/ReadVariableOpReadVariableOp'linear/linear_model/dense_input/weights* 
_output_shapes
:
??*
dtype0
a
zero_fraction/total_size/SizeConst*
_output_shapes
: *
dtype0	*
valueB		 R??
`
zero_fraction/total_zero/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
zero_fraction/total_zero/EqualEqualzero_fraction/total_size/Sizezero_fraction/total_zero/Const*
T0	*
_output_shapes
: 
?
#zero_fraction/total_zero/zero_countIfzero_fraction/total_zero/Equal'linear/linear_model/dense_input/weightszero_fraction/total_size/Size*
Tcond0
*
Tin
2	*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *A
else_branch2R0
.zero_fraction_total_zero_zero_count_false_1288*
output_shapes
: *@
then_branch1R/
-zero_fraction_total_zero_zero_count_true_1287
~
,zero_fraction/total_zero/zero_count/IdentityIdentity#zero_fraction/total_zero/zero_count*
T0*
_output_shapes
: 
y
"zero_fraction/compute/float32_sizeCastzero_fraction/total_size/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
zero_fraction/compute/truedivRealDiv,zero_fraction/total_zero/zero_count/Identity"zero_fraction/compute/float32_size*
T0*
_output_shapes
: 
n
"zero_fraction/zero_fraction_or_nanIdentityzero_fraction/compute/truediv*
T0*
_output_shapes
: 
v
fraction_of_zero_weights/tagsConst*
_output_shapes
: *
dtype0*)
value B Bfraction_of_zero_weights
?
fraction_of_zero_weightsScalarSummaryfraction_of_zero_weights/tags"zero_fraction/zero_fraction_or_nan*
T0*
_output_shapes
: 
?
head/logits/ShapeShapeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*
_output_shapes
:
g
%head/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :
W
Ohead/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
H
@head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
?
head/predictions/logisticSigmoidHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*'
_output_shapes
:?????????
?
head/predictions/zeros_like	ZerosLikeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*'
_output_shapes
:?????????
q
&head/predictions/two_class_logits/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
!head/predictions/two_class_logitsConcatV2head/predictions/zeros_likeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum&head/predictions/two_class_logits/axis*
N*
T0*'
_output_shapes
:?????????
~
head/predictions/probabilitiesSoftmax!head/predictions/two_class_logits*
T0*'
_output_shapes
:?????????
o
$head/predictions/class_ids/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
head/predictions/class_idsArgMax!head/predictions/two_class_logits$head/predictions/class_ids/dimension*
T0*#
_output_shapes
:?????????
j
head/predictions/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
head/predictions/ExpandDims
ExpandDimshead/predictions/class_idshead/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:?????????
w
head/predictions/str_classesAsStringhead/predictions/ExpandDims*
T0	*'
_output_shapes
:?????????
?
head/predictions/ShapeShapeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*
_output_shapes
:
n
$head/predictions/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
p
&head/predictions/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
p
&head/predictions/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
head/predictions/strided_sliceStridedSlicehead/predictions/Shape$head/predictions/strided_slice/stack&head/predictions/strided_slice/stack_1&head/predictions/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
^
head/predictions/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
^
head/predictions/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
^
head/predictions/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/rangeRangehead/predictions/range/starthead/predictions/range/limithead/predictions/range/delta*
_output_shapes
:
c
!head/predictions/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
head/predictions/ExpandDims_1
ExpandDimshead/predictions/range!head/predictions/ExpandDims_1/dim*
T0*
_output_shapes

:
c
!head/predictions/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/Tile/multiplesPackhead/predictions/strided_slice!head/predictions/Tile/multiples/1*
N*
T0*
_output_shapes
:
?
head/predictions/TileTilehead/predictions/ExpandDims_1head/predictions/Tile/multiples*
T0*'
_output_shapes
:?????????
?
head/predictions/Shape_1ShapeHlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum*
T0*
_output_shapes
:
p
&head/predictions/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
r
(head/predictions/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
r
(head/predictions/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
 head/predictions/strided_slice_1StridedSlicehead/predictions/Shape_1&head/predictions/strided_slice_1/stack(head/predictions/strided_slice_1/stack_1(head/predictions/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
`
head/predictions/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
`
head/predictions/range_1/limitConst*
_output_shapes
: *
dtype0*
value	B :
`
head/predictions/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/range_1Rangehead/predictions/range_1/starthead/predictions/range_1/limithead/predictions/range_1/delta*
_output_shapes
:
d
head/predictions/AsStringAsStringhead/predictions/range_1*
T0*
_output_shapes
:
c
!head/predictions/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
head/predictions/ExpandDims_2
ExpandDimshead/predictions/AsString!head/predictions/ExpandDims_2/dim*
T0*
_output_shapes

:
e
#head/predictions/Tile_1/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
?
!head/predictions/Tile_1/multiplesPack head/predictions/strided_slice_1#head/predictions/Tile_1/multiples/1*
N*
T0*
_output_shapes
:
?
head/predictions/Tile_1Tilehead/predictions/ExpandDims_2!head/predictions/Tile_1/multiples*
T0*'
_output_shapes
:?????????
X

head/ShapeShapehead/predictions/probabilities*
T0*
_output_shapes
:
b
head/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
d
head/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
d
head/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
head/strided_sliceStridedSlice
head/Shapehead/strided_slice/stackhead/strided_slice/stack_1head/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
R
head/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
R
head/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
R
head/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
e

head/rangeRangehead/range/starthead/range/limithead/range/delta*
_output_shapes
:
J
head/AsStringAsString
head/range*
T0*
_output_shapes
:
U
head/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
j
head/ExpandDims
ExpandDimshead/AsStringhead/ExpandDims/dim*
T0*
_output_shapes

:
W
head/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
t
head/Tile/multiplesPackhead/strided_slicehead/Tile/multiples/1*
N*
T0*
_output_shapes
:
i
	head/TileTilehead/ExpandDimshead/Tile/multiples*
T0*'
_output_shapes
:?????????

initNoOp
?
init_all_tablesNoOpj^transform/transform/compute_and_apply_vocabulary/apply_vocab/text_file_init/InitializeTableFromTextFileV2

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
?
save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5fa3e0267a55413b8ab4d2d11ffac941/part
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*k
valuebB`Bglobal_stepB linear/linear_model/bias_weightsB'linear/linear_model/dense_input/weights
x
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step/Read/ReadVariableOp4linear/linear_model/bias_weights/Read/ReadVariableOp;linear/linear_model/dense_input/weights/Read/ReadVariableOp"/device:CPU:0*
dtypes
2	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*k
valuebB`Bglobal_stepB linear/linear_model/bias_weightsB'linear/linear_model/dense_input/weights
{
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0* 
_output_shapes
:::*
dtypes
2	
N
save/Identity_1Identitysave/RestoreV2*
T0	*
_output_shapes
:
T
save/AssignVariableOpAssignVariableOpglobal_stepsave/Identity_1*
dtype0	
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
k
save/AssignVariableOp_1AssignVariableOp linear/linear_model/bias_weightssave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
r
save/AssignVariableOp_2AssignVariableOp'linear/linear_model/dense_input/weightssave/Identity_3*
dtype0
f
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2
-
save/restore_allNoOp^save/restore_shard?#
?
^
-zero_fraction_total_zero_zero_count_true_1287
placeholder
placeholder_1		
constS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const"
constConst:output:0*
_input_shapes
:: 
?
?
.zero_fraction_total_zero_zero_count_false_1288H
Dzero_fraction_readvariableop_linear_linear_model_dense_input_weights&
"cast_zero_fraction_total_size_size	
mul??
zero_fraction/ReadVariableOpReadVariableOpDzero_fraction_readvariableop_linear_linear_model_dense_input_weights* 
_output_shapes
:
??*
dtype02
zero_fraction/ReadVariableOpl
zero_fraction/SizeConst*
_output_shapes
: *
dtype0	*
valueB		 R??2
zero_fraction/Size|
zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????2
zero_fraction/LessEqual/y?
zero_fraction/LessEqual	LessEqualzero_fraction/Size:output:0"zero_fraction/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
zero_fraction/LessEqual?
zero_fraction/condStatelessIfzero_fraction/LessEqual:z:0$zero_fraction/ReadVariableOp:value:0*
Tcond0
*
Tin
2*
Tout
2	*
_lower_using_switch_merge(*
_output_shapes
: *0
else_branch!R
zero_fraction_cond_false_1298*
output_shapes
: */
then_branch R
zero_fraction_cond_true_12972
zero_fraction/cond?
zero_fraction/cond/IdentityIdentityzero_fraction/cond:output:0*
T0	*
_output_shapes
: 2
zero_fraction/cond/Identity?
$zero_fraction/counts_to_fraction/subSubzero_fraction/Size:output:0$zero_fraction/cond/Identity:output:0*
T0	*
_output_shapes
: 2&
$zero_fraction/counts_to_fraction/sub?
%zero_fraction/counts_to_fraction/CastCast(zero_fraction/counts_to_fraction/sub:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2'
%zero_fraction/counts_to_fraction/Cast?
'zero_fraction/counts_to_fraction/Cast_1Castzero_fraction/Size:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2)
'zero_fraction/counts_to_fraction/Cast_1?
(zero_fraction/counts_to_fraction/truedivRealDiv)zero_fraction/counts_to_fraction/Cast:y:0+zero_fraction/counts_to_fraction/Cast_1:y:0*
T0*
_output_shapes
: 2*
(zero_fraction/counts_to_fraction/truediv?
zero_fraction/fractionIdentity,zero_fraction/counts_to_fraction/truediv:z:0*
T0*
_output_shapes
: 2
zero_fraction/fractionh
CastCast"cast_zero_fraction_total_size_size*

DstT0*

SrcT0	*
_output_shapes
: 2
CastG
mul_0Mulzero_fraction/fraction:output:0Cast:y:0*
T02
mul"
mul	mul_0:z:0*
_input_shapes
:: 
?
a
zero_fraction_cond_true_12977
3count_nonzero_notequal_zero_fraction_readvariableop
cast	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0* 
_output_shapes
:
??2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0*

SrcT0
* 
_output_shapes
:
??2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: 2
count_nonzero/nonzero_countj
CastCast$count_nonzero/nonzero_count:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast"
castCast:y:0*
_input_shapes
:
??
?
y
zero_fraction_cond_false_12987
3count_nonzero_notequal_zero_fraction_readvariableop
count_nonzero_nonzero_count	o
count_nonzero/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2
count_nonzero/zeros?
count_nonzero/NotEqualNotEqual3count_nonzero_notequal_zero_fraction_readvariableopcount_nonzero/zeros:output:0*
T0* 
_output_shapes
:
??2
count_nonzero/NotEqual?
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

DstT0	*

SrcT0
* 
_output_shapes
:
??2
count_nonzero/Cast{
count_nonzero/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
count_nonzero/Const?
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0	*
_output_shapes
: 2
count_nonzero/nonzero_count"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*
_input_shapes
:
??"?<
save/Const:0save/Identity:0save/restore_all (5 @F8"
asset_filepaths
	
Const:0"~
global_stepom
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H"?
saved_model_assetsm*k
i
+type.googleapis.com/tensorflow.AssetFileDef:
	
Const:0-vocab_compute_and_apply_vocabulary_vocabulary"%
saved_model_main_op


group_deps"3
	summaries&
$
bias:0
fraction_of_zero_weights:0"?
table_initializerm
k
itransform/transform/compute_and_apply_vocabulary/apply_vocab/text_file_init/InitializeTableFromTextFileV2"e
tft_schema_override_maxJ
H
Ftransform/transform/compute_and_apply_vocabulary/apply_vocab/Maximum:0"e
tft_schema_override_minJ
H
Ftransform/transform/compute_and_apply_vocabulary/apply_vocab/Minimum:0"?
tft_schema_override_tensorf
d
btransform/transform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/LookupTableFindV2:0"?
trainable_variables??
?
)linear/linear_model/dense_input/weights:0.linear/linear_model/dense_input/weights/Assign=linear/linear_model/dense_input/weights/Read/ReadVariableOp:0(2;linear/linear_model/dense_input/weights/Initializer/zeros:08
?
"linear/linear_model/bias_weights:0'linear/linear_model/bias_weights/Assign6linear/linear_model/bias_weights/Read/ReadVariableOp:0(24linear/linear_model/bias_weights/Initializer/zeros:08"?
	variables??
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H
?
)linear/linear_model/dense_input/weights:0.linear/linear_model/dense_input/weights/Assign=linear/linear_model/dense_input/weights/Read/ReadVariableOp:0(2;linear/linear_model/dense_input/weights/Initializer/zeros:08
?
"linear/linear_model/bias_weights:0'linear/linear_model/bias_weights/Assign6linear/linear_model/bias_weights/Read/ReadVariableOp:0(24linear/linear_model/bias_weights/Initializer/zeros:08*?
classification?
3
inputs)
input_example_tensor:0?????????-
classes"
head/Tile:0?????????A
scores7
 head/predictions/probabilities:0?????????tensorflow/serving/classify*?
predict?
5
examples)
input_example_tensor:0??????????
all_class_ids.
head/predictions/Tile:0??????????
all_classes0
head/predictions/Tile_1:0?????????A
	class_ids4
head/predictions/ExpandDims:0	?????????@
classes5
head/predictions/str_classes:0?????????>
logistic2
head/predictions/logistic:0?????????k
logitsa
Jlinear/linear_model/linear/linear_model/linear/linear_model/weighted_sum:0?????????H
probabilities7
 head/predictions/probabilities:0?????????tensorflow/serving/predict*?

regression?
3
inputs)
input_example_tensor:0?????????=
outputs2
head/predictions/logistic:0?????????tensorflow/serving/regress*?
serving_default?
3
inputs)
input_example_tensor:0?????????-
classes"
head/Tile:0?????????A
scores7
 head/predictions/probabilities:0?????????tensorflow/serving/classify