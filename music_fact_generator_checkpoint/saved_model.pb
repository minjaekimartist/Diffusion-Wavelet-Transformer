õ
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
*
Erf
x"T
y"T"
Ttype:
2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
6Adam/v/transformer_denoiser/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86Adam/v/transformer_denoiser/layer_normalization_1/beta
�
JAdam/v/transformer_denoiser/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp6Adam/v/transformer_denoiser/layer_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
6Adam/m/transformer_denoiser/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86Adam/m/transformer_denoiser/layer_normalization_1/beta
�
JAdam/m/transformer_denoiser/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp6Adam/m/transformer_denoiser/layer_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
7Adam/v/transformer_denoiser/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*H
shared_name97Adam/v/transformer_denoiser/layer_normalization_1/gamma
�
KAdam/v/transformer_denoiser/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp7Adam/v/transformer_denoiser/layer_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
7Adam/m/transformer_denoiser/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*H
shared_name97Adam/m/transformer_denoiser/layer_normalization_1/gamma
�
KAdam/m/transformer_denoiser/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp7Adam/m/transformer_denoiser/layer_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
4Adam/v/transformer_denoiser/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64Adam/v/transformer_denoiser/layer_normalization/beta
�
HAdam/v/transformer_denoiser/layer_normalization/beta/Read/ReadVariableOpReadVariableOp4Adam/v/transformer_denoiser/layer_normalization/beta*
_output_shapes	
:�*
dtype0
�
4Adam/m/transformer_denoiser/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64Adam/m/transformer_denoiser/layer_normalization/beta
�
HAdam/m/transformer_denoiser/layer_normalization/beta/Read/ReadVariableOpReadVariableOp4Adam/m/transformer_denoiser/layer_normalization/beta*
_output_shapes	
:�*
dtype0
�
5Adam/v/transformer_denoiser/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75Adam/v/transformer_denoiser/layer_normalization/gamma
�
IAdam/v/transformer_denoiser/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp5Adam/v/transformer_denoiser/layer_normalization/gamma*
_output_shapes	
:�*
dtype0
�
5Adam/m/transformer_denoiser/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*F
shared_name75Adam/m/transformer_denoiser/layer_normalization/gamma
�
IAdam/m/transformer_denoiser/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp5Adam/m/transformer_denoiser/layer_normalization/gamma*
_output_shapes	
:�*
dtype0
�
(Adam/v/transformer_denoiser/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/v/transformer_denoiser/dense_4/bias
�
<Adam/v/transformer_denoiser/dense_4/bias/Read/ReadVariableOpReadVariableOp(Adam/v/transformer_denoiser/dense_4/bias*
_output_shapes	
:�*
dtype0
�
(Adam/m/transformer_denoiser/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/m/transformer_denoiser/dense_4/bias
�
<Adam/m/transformer_denoiser/dense_4/bias/Read/ReadVariableOpReadVariableOp(Adam/m/transformer_denoiser/dense_4/bias*
_output_shapes	
:�*
dtype0
�
*Adam/v/transformer_denoiser/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/v/transformer_denoiser/dense_4/kernel
�
>Adam/v/transformer_denoiser/dense_4/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/transformer_denoiser/dense_4/kernel* 
_output_shapes
:
��*
dtype0
�
*Adam/m/transformer_denoiser/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/m/transformer_denoiser/dense_4/kernel
�
>Adam/m/transformer_denoiser/dense_4/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/transformer_denoiser/dense_4/kernel* 
_output_shapes
:
��*
dtype0
�
(Adam/v/transformer_denoiser/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/v/transformer_denoiser/dense_3/bias
�
<Adam/v/transformer_denoiser/dense_3/bias/Read/ReadVariableOpReadVariableOp(Adam/v/transformer_denoiser/dense_3/bias*
_output_shapes	
:�*
dtype0
�
(Adam/m/transformer_denoiser/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/m/transformer_denoiser/dense_3/bias
�
<Adam/m/transformer_denoiser/dense_3/bias/Read/ReadVariableOpReadVariableOp(Adam/m/transformer_denoiser/dense_3/bias*
_output_shapes	
:�*
dtype0
�
*Adam/v/transformer_denoiser/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/v/transformer_denoiser/dense_3/kernel
�
>Adam/v/transformer_denoiser/dense_3/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/transformer_denoiser/dense_3/kernel* 
_output_shapes
:
��*
dtype0
�
*Adam/m/transformer_denoiser/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/m/transformer_denoiser/dense_3/kernel
�
>Adam/m/transformer_denoiser/dense_3/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/transformer_denoiser/dense_3/kernel* 
_output_shapes
:
��*
dtype0
�
FAdam/v/transformer_denoiser/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*W
shared_nameHFAdam/v/transformer_denoiser/multi_head_attention/attention_output/bias
�
ZAdam/v/transformer_denoiser/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOpFAdam/v/transformer_denoiser/multi_head_attention/attention_output/bias*
_output_shapes	
:�*
dtype0
�
FAdam/m/transformer_denoiser/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*W
shared_nameHFAdam/m/transformer_denoiser/multi_head_attention/attention_output/bias
�
ZAdam/m/transformer_denoiser/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOpFAdam/m/transformer_denoiser/multi_head_attention/attention_output/bias*
_output_shapes	
:�*
dtype0
�
HAdam/v/transformer_denoiser/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*Y
shared_nameJHAdam/v/transformer_denoiser/multi_head_attention/attention_output/kernel
�
\Adam/v/transformer_denoiser/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpHAdam/v/transformer_denoiser/multi_head_attention/attention_output/kernel*#
_output_shapes
:@�*
dtype0
�
HAdam/m/transformer_denoiser/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*Y
shared_nameJHAdam/m/transformer_denoiser/multi_head_attention/attention_output/kernel
�
\Adam/m/transformer_denoiser/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpHAdam/m/transformer_denoiser/multi_head_attention/attention_output/kernel*#
_output_shapes
:@�*
dtype0
�
;Adam/v/transformer_denoiser/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*L
shared_name=;Adam/v/transformer_denoiser/multi_head_attention/value/bias
�
OAdam/v/transformer_denoiser/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp;Adam/v/transformer_denoiser/multi_head_attention/value/bias*
_output_shapes

:@*
dtype0
�
;Adam/m/transformer_denoiser/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*L
shared_name=;Adam/m/transformer_denoiser/multi_head_attention/value/bias
�
OAdam/m/transformer_denoiser/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp;Adam/m/transformer_denoiser/multi_head_attention/value/bias*
_output_shapes

:@*
dtype0
�
=Adam/v/transformer_denoiser/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*N
shared_name?=Adam/v/transformer_denoiser/multi_head_attention/value/kernel
�
QAdam/v/transformer_denoiser/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp=Adam/v/transformer_denoiser/multi_head_attention/value/kernel*#
_output_shapes
:�@*
dtype0
�
=Adam/m/transformer_denoiser/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*N
shared_name?=Adam/m/transformer_denoiser/multi_head_attention/value/kernel
�
QAdam/m/transformer_denoiser/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp=Adam/m/transformer_denoiser/multi_head_attention/value/kernel*#
_output_shapes
:�@*
dtype0
�
9Adam/v/transformer_denoiser/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*J
shared_name;9Adam/v/transformer_denoiser/multi_head_attention/key/bias
�
MAdam/v/transformer_denoiser/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp9Adam/v/transformer_denoiser/multi_head_attention/key/bias*
_output_shapes

:@*
dtype0
�
9Adam/m/transformer_denoiser/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*J
shared_name;9Adam/m/transformer_denoiser/multi_head_attention/key/bias
�
MAdam/m/transformer_denoiser/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp9Adam/m/transformer_denoiser/multi_head_attention/key/bias*
_output_shapes

:@*
dtype0
�
;Adam/v/transformer_denoiser/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*L
shared_name=;Adam/v/transformer_denoiser/multi_head_attention/key/kernel
�
OAdam/v/transformer_denoiser/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp;Adam/v/transformer_denoiser/multi_head_attention/key/kernel*#
_output_shapes
:�@*
dtype0
�
;Adam/m/transformer_denoiser/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*L
shared_name=;Adam/m/transformer_denoiser/multi_head_attention/key/kernel
�
OAdam/m/transformer_denoiser/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp;Adam/m/transformer_denoiser/multi_head_attention/key/kernel*#
_output_shapes
:�@*
dtype0
�
;Adam/v/transformer_denoiser/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*L
shared_name=;Adam/v/transformer_denoiser/multi_head_attention/query/bias
�
OAdam/v/transformer_denoiser/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp;Adam/v/transformer_denoiser/multi_head_attention/query/bias*
_output_shapes

:@*
dtype0
�
;Adam/m/transformer_denoiser/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*L
shared_name=;Adam/m/transformer_denoiser/multi_head_attention/query/bias
�
OAdam/m/transformer_denoiser/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp;Adam/m/transformer_denoiser/multi_head_attention/query/bias*
_output_shapes

:@*
dtype0
�
=Adam/v/transformer_denoiser/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*N
shared_name?=Adam/v/transformer_denoiser/multi_head_attention/query/kernel
�
QAdam/v/transformer_denoiser/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp=Adam/v/transformer_denoiser/multi_head_attention/query/kernel*#
_output_shapes
:�@*
dtype0
�
=Adam/m/transformer_denoiser/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*N
shared_name?=Adam/m/transformer_denoiser/multi_head_attention/query/kernel
�
QAdam/m/transformer_denoiser/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp=Adam/m/transformer_denoiser/multi_head_attention/query/kernel*#
_output_shapes
:�@*
dtype0
�
(Adam/v/transformer_denoiser/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/v/transformer_denoiser/dense_2/bias
�
<Adam/v/transformer_denoiser/dense_2/bias/Read/ReadVariableOpReadVariableOp(Adam/v/transformer_denoiser/dense_2/bias*
_output_shapes	
:�*
dtype0
�
(Adam/m/transformer_denoiser/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/m/transformer_denoiser/dense_2/bias
�
<Adam/m/transformer_denoiser/dense_2/bias/Read/ReadVariableOpReadVariableOp(Adam/m/transformer_denoiser/dense_2/bias*
_output_shapes	
:�*
dtype0
�
*Adam/v/transformer_denoiser/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/v/transformer_denoiser/dense_2/kernel
�
>Adam/v/transformer_denoiser/dense_2/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/transformer_denoiser/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
*Adam/m/transformer_denoiser/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/m/transformer_denoiser/dense_2/kernel
�
>Adam/m/transformer_denoiser/dense_2/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/transformer_denoiser/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
(Adam/v/transformer_denoiser/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/v/transformer_denoiser/dense_1/bias
�
<Adam/v/transformer_denoiser/dense_1/bias/Read/ReadVariableOpReadVariableOp(Adam/v/transformer_denoiser/dense_1/bias*
_output_shapes	
:�*
dtype0
�
(Adam/m/transformer_denoiser/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/m/transformer_denoiser/dense_1/bias
�
<Adam/m/transformer_denoiser/dense_1/bias/Read/ReadVariableOpReadVariableOp(Adam/m/transformer_denoiser/dense_1/bias*
_output_shapes	
:�*
dtype0
�
*Adam/v/transformer_denoiser/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/v/transformer_denoiser/dense_1/kernel
�
>Adam/v/transformer_denoiser/dense_1/kernel/Read/ReadVariableOpReadVariableOp*Adam/v/transformer_denoiser/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
*Adam/m/transformer_denoiser/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/m/transformer_denoiser/dense_1/kernel
�
>Adam/m/transformer_denoiser/dense_1/kernel/Read/ReadVariableOpReadVariableOp*Adam/m/transformer_denoiser/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
&Adam/v/transformer_denoiser/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&Adam/v/transformer_denoiser/dense/bias
�
:Adam/v/transformer_denoiser/dense/bias/Read/ReadVariableOpReadVariableOp&Adam/v/transformer_denoiser/dense/bias*
_output_shapes	
:�*
dtype0
�
&Adam/m/transformer_denoiser/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&Adam/m/transformer_denoiser/dense/bias
�
:Adam/m/transformer_denoiser/dense/bias/Read/ReadVariableOpReadVariableOp&Adam/m/transformer_denoiser/dense/bias*
_output_shapes	
:�*
dtype0
�
(Adam/v/transformer_denoiser/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*9
shared_name*(Adam/v/transformer_denoiser/dense/kernel
�
<Adam/v/transformer_denoiser/dense/kernel/Read/ReadVariableOpReadVariableOp(Adam/v/transformer_denoiser/dense/kernel* 
_output_shapes
:
��*
dtype0
�
(Adam/m/transformer_denoiser/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*9
shared_name*(Adam/m/transformer_denoiser/dense/kernel
�
<Adam/m/transformer_denoiser/dense/kernel/Read/ReadVariableOpReadVariableOp(Adam/m/transformer_denoiser/dense/kernel* 
_output_shapes
:
��*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
/transformer_denoiser/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*@
shared_name1/transformer_denoiser/layer_normalization_1/beta
�
Ctransformer_denoiser/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp/transformer_denoiser/layer_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
0transformer_denoiser/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20transformer_denoiser/layer_normalization_1/gamma
�
Dtransformer_denoiser/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp0transformer_denoiser/layer_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
-transformer_denoiser/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-transformer_denoiser/layer_normalization/beta
�
Atransformer_denoiser/layer_normalization/beta/Read/ReadVariableOpReadVariableOp-transformer_denoiser/layer_normalization/beta*
_output_shapes	
:�*
dtype0
�
.transformer_denoiser/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.transformer_denoiser/layer_normalization/gamma
�
Btransformer_denoiser/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp.transformer_denoiser/layer_normalization/gamma*
_output_shapes	
:�*
dtype0
�
!transformer_denoiser/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!transformer_denoiser/dense_4/bias
�
5transformer_denoiser/dense_4/bias/Read/ReadVariableOpReadVariableOp!transformer_denoiser/dense_4/bias*
_output_shapes	
:�*
dtype0
�
#transformer_denoiser/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#transformer_denoiser/dense_4/kernel
�
7transformer_denoiser/dense_4/kernel/Read/ReadVariableOpReadVariableOp#transformer_denoiser/dense_4/kernel* 
_output_shapes
:
��*
dtype0
�
!transformer_denoiser/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!transformer_denoiser/dense_3/bias
�
5transformer_denoiser/dense_3/bias/Read/ReadVariableOpReadVariableOp!transformer_denoiser/dense_3/bias*
_output_shapes	
:�*
dtype0
�
#transformer_denoiser/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#transformer_denoiser/dense_3/kernel
�
7transformer_denoiser/dense_3/kernel/Read/ReadVariableOpReadVariableOp#transformer_denoiser/dense_3/kernel* 
_output_shapes
:
��*
dtype0
�
?transformer_denoiser/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*P
shared_nameA?transformer_denoiser/multi_head_attention/attention_output/bias
�
Stransformer_denoiser/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp?transformer_denoiser/multi_head_attention/attention_output/bias*
_output_shapes	
:�*
dtype0
�
Atransformer_denoiser/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*R
shared_nameCAtransformer_denoiser/multi_head_attention/attention_output/kernel
�
Utransformer_denoiser/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpAtransformer_denoiser/multi_head_attention/attention_output/kernel*#
_output_shapes
:@�*
dtype0
�
4transformer_denoiser/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*E
shared_name64transformer_denoiser/multi_head_attention/value/bias
�
Htransformer_denoiser/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp4transformer_denoiser/multi_head_attention/value/bias*
_output_shapes

:@*
dtype0
�
6transformer_denoiser/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*G
shared_name86transformer_denoiser/multi_head_attention/value/kernel
�
Jtransformer_denoiser/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp6transformer_denoiser/multi_head_attention/value/kernel*#
_output_shapes
:�@*
dtype0
�
2transformer_denoiser/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*C
shared_name42transformer_denoiser/multi_head_attention/key/bias
�
Ftransformer_denoiser/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp2transformer_denoiser/multi_head_attention/key/bias*
_output_shapes

:@*
dtype0
�
4transformer_denoiser/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*E
shared_name64transformer_denoiser/multi_head_attention/key/kernel
�
Htransformer_denoiser/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp4transformer_denoiser/multi_head_attention/key/kernel*#
_output_shapes
:�@*
dtype0
�
4transformer_denoiser/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*E
shared_name64transformer_denoiser/multi_head_attention/query/bias
�
Htransformer_denoiser/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp4transformer_denoiser/multi_head_attention/query/bias*
_output_shapes

:@*
dtype0
�
6transformer_denoiser/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*G
shared_name86transformer_denoiser/multi_head_attention/query/kernel
�
Jtransformer_denoiser/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp6transformer_denoiser/multi_head_attention/query/kernel*#
_output_shapes
:�@*
dtype0
�
!transformer_denoiser/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!transformer_denoiser/dense_2/bias
�
5transformer_denoiser/dense_2/bias/Read/ReadVariableOpReadVariableOp!transformer_denoiser/dense_2/bias*
_output_shapes	
:�*
dtype0
�
#transformer_denoiser/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#transformer_denoiser/dense_2/kernel
�
7transformer_denoiser/dense_2/kernel/Read/ReadVariableOpReadVariableOp#transformer_denoiser/dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
!transformer_denoiser/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!transformer_denoiser/dense_1/bias
�
5transformer_denoiser/dense_1/bias/Read/ReadVariableOpReadVariableOp!transformer_denoiser/dense_1/bias*
_output_shapes	
:�*
dtype0
�
#transformer_denoiser/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#transformer_denoiser/dense_1/kernel
�
7transformer_denoiser/dense_1/kernel/Read/ReadVariableOpReadVariableOp#transformer_denoiser/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
transformer_denoiser/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!transformer_denoiser/dense/bias
�
3transformer_denoiser/dense/bias/Read/ReadVariableOpReadVariableOptransformer_denoiser/dense/bias*
_output_shapes	
:�*
dtype0
�
!transformer_denoiser/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*2
shared_name#!transformer_denoiser/dense/kernel
�
5transformer_denoiser/dense/kernel/Read/ReadVariableOpReadVariableOp!transformer_denoiser/dense/kernel* 
_output_shapes
:
��*
dtype0
|
serving_default_input_1Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1.transformer_denoiser/layer_normalization/gamma-transformer_denoiser/layer_normalization/beta!transformer_denoiser/dense/kerneltransformer_denoiser/dense/bias#transformer_denoiser/dense_1/kernel!transformer_denoiser/dense_1/bias#transformer_denoiser/dense_2/kernel!transformer_denoiser/dense_2/bias6transformer_denoiser/multi_head_attention/query/kernel4transformer_denoiser/multi_head_attention/query/bias4transformer_denoiser/multi_head_attention/key/kernel2transformer_denoiser/multi_head_attention/key/bias6transformer_denoiser/multi_head_attention/value/kernel4transformer_denoiser/multi_head_attention/value/biasAtransformer_denoiser/multi_head_attention/attention_output/kernel?transformer_denoiser/multi_head_attention/attention_output/bias0transformer_denoiser/layer_normalization_1/gamma/transformer_denoiser/layer_normalization_1/beta#transformer_denoiser/dense_3/kernel!transformer_denoiser/dense_3/bias#transformer_denoiser/dense_4/kernel!transformer_denoiser/dense_4/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_217358

NoOpNoOp
Ӝ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2
	
proj1
	attention

dense3

dense4
	norm1
	norm2
dropout1
dropout2
	optimizer

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21*
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21*
* 
�
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

/trace_0
0trace_1* 

1trace_0
2trace_1* 
* 
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

kernel
bias*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

kernel
bias*
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K_query_dense
L
_key_dense
M_value_dense
N_softmax
O_dropout_layer
P_output_dense*
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

"kernel
#bias*
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

$kernel
%bias*
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
caxis
	&gamma
'beta*
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
jaxis
	(gamma
)beta*
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
q_random_generator* 
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
x_random_generator* 
�
y
_variables
z_iterations
{_learning_rate
|_index_dict
}
_momentums
~_velocities
_update_step_xla*

�serving_default* 
a[
VARIABLE_VALUE!transformer_denoiser/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEtransformer_denoiser/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#transformer_denoiser/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!transformer_denoiser/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#transformer_denoiser/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!transformer_denoiser/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6transformer_denoiser/multi_head_attention/query/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4transformer_denoiser/multi_head_attention/query/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4transformer_denoiser/multi_head_attention/key/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE2transformer_denoiser/multi_head_attention/key/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6transformer_denoiser/multi_head_attention/value/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4transformer_denoiser/multi_head_attention/value/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAtransformer_denoiser/multi_head_attention/attention_output/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE?transformer_denoiser/multi_head_attention/attention_output/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#transformer_denoiser/dense_3/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!transformer_denoiser/dense_3/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#transformer_denoiser/dense_4/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!transformer_denoiser/dense_4/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.transformer_denoiser/layer_normalization/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-transformer_denoiser/layer_normalization/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0transformer_denoiser/layer_normalization_1/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_denoiser/layer_normalization_1/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
* 
J
0
	1

2
3
4
5
6
7
8
9*

�0*
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
<
0
1
2
3
4
5
 6
!7*
<
0
1
2
3
4
5
 6
!7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

 kernel
!bias*

"0
#1*

"0
#1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

$0
%1*

$0
%1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

&0
'1*

&0
'1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

(0
)1*

(0
)1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
z0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21*
* 
* 
<
�	variables
�	keras_api

�total

�count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
.
K0
L1
M2
N3
O4
P5*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

 0
!1*

 0
!1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
sm
VARIABLE_VALUE(Adam/m/transformer_denoiser/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/transformer_denoiser/dense/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/m/transformer_denoiser/dense/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/v/transformer_denoiser/dense/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/transformer_denoiser/dense_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/v/transformer_denoiser/dense_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/m/transformer_denoiser/dense_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE(Adam/v/transformer_denoiser/dense_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/transformer_denoiser/dense_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/transformer_denoiser/dense_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/transformer_denoiser/dense_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/transformer_denoiser/dense_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE=Adam/m/transformer_denoiser/multi_head_attention/query/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE=Adam/v/transformer_denoiser/multi_head_attention/query/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE;Adam/m/transformer_denoiser/multi_head_attention/query/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE;Adam/v/transformer_denoiser/multi_head_attention/query/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE;Adam/m/transformer_denoiser/multi_head_attention/key/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE;Adam/v/transformer_denoiser/multi_head_attention/key/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE9Adam/m/transformer_denoiser/multi_head_attention/key/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE9Adam/v/transformer_denoiser/multi_head_attention/key/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE=Adam/m/transformer_denoiser/multi_head_attention/value/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE=Adam/v/transformer_denoiser/multi_head_attention/value/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE;Adam/m/transformer_denoiser/multi_head_attention/value/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE;Adam/v/transformer_denoiser/multi_head_attention/value/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEHAdam/m/transformer_denoiser/multi_head_attention/attention_output/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEHAdam/v/transformer_denoiser/multi_head_attention/attention_output/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEFAdam/m/transformer_denoiser/multi_head_attention/attention_output/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEFAdam/v/transformer_denoiser/multi_head_attention/attention_output/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/transformer_denoiser/dense_3/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/transformer_denoiser/dense_3/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/transformer_denoiser/dense_3/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/transformer_denoiser/dense_3/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/m/transformer_denoiser/dense_4/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE*Adam/v/transformer_denoiser/dense_4/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/m/transformer_denoiser/dense_4/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE(Adam/v/transformer_denoiser/dense_4/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/m/transformer_denoiser/layer_normalization/gamma2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE5Adam/v/transformer_denoiser/layer_normalization/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4Adam/m/transformer_denoiser/layer_normalization/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4Adam/v/transformer_denoiser/layer_normalization/beta2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUE7Adam/m/transformer_denoiser/layer_normalization_1/gamma2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUE7Adam/v/transformer_denoiser/layer_normalization_1/gamma2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE6Adam/m/transformer_denoiser/layer_normalization_1/beta2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE6Adam/v/transformer_denoiser/layer_normalization_1/beta2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!transformer_denoiser/dense/kerneltransformer_denoiser/dense/bias#transformer_denoiser/dense_1/kernel!transformer_denoiser/dense_1/bias#transformer_denoiser/dense_2/kernel!transformer_denoiser/dense_2/bias6transformer_denoiser/multi_head_attention/query/kernel4transformer_denoiser/multi_head_attention/query/bias4transformer_denoiser/multi_head_attention/key/kernel2transformer_denoiser/multi_head_attention/key/bias6transformer_denoiser/multi_head_attention/value/kernel4transformer_denoiser/multi_head_attention/value/biasAtransformer_denoiser/multi_head_attention/attention_output/kernel?transformer_denoiser/multi_head_attention/attention_output/bias#transformer_denoiser/dense_3/kernel!transformer_denoiser/dense_3/bias#transformer_denoiser/dense_4/kernel!transformer_denoiser/dense_4/bias.transformer_denoiser/layer_normalization/gamma-transformer_denoiser/layer_normalization/beta0transformer_denoiser/layer_normalization_1/gamma/transformer_denoiser/layer_normalization_1/beta	iterationlearning_rate(Adam/m/transformer_denoiser/dense/kernel(Adam/v/transformer_denoiser/dense/kernel&Adam/m/transformer_denoiser/dense/bias&Adam/v/transformer_denoiser/dense/bias*Adam/m/transformer_denoiser/dense_1/kernel*Adam/v/transformer_denoiser/dense_1/kernel(Adam/m/transformer_denoiser/dense_1/bias(Adam/v/transformer_denoiser/dense_1/bias*Adam/m/transformer_denoiser/dense_2/kernel*Adam/v/transformer_denoiser/dense_2/kernel(Adam/m/transformer_denoiser/dense_2/bias(Adam/v/transformer_denoiser/dense_2/bias=Adam/m/transformer_denoiser/multi_head_attention/query/kernel=Adam/v/transformer_denoiser/multi_head_attention/query/kernel;Adam/m/transformer_denoiser/multi_head_attention/query/bias;Adam/v/transformer_denoiser/multi_head_attention/query/bias;Adam/m/transformer_denoiser/multi_head_attention/key/kernel;Adam/v/transformer_denoiser/multi_head_attention/key/kernel9Adam/m/transformer_denoiser/multi_head_attention/key/bias9Adam/v/transformer_denoiser/multi_head_attention/key/bias=Adam/m/transformer_denoiser/multi_head_attention/value/kernel=Adam/v/transformer_denoiser/multi_head_attention/value/kernel;Adam/m/transformer_denoiser/multi_head_attention/value/bias;Adam/v/transformer_denoiser/multi_head_attention/value/biasHAdam/m/transformer_denoiser/multi_head_attention/attention_output/kernelHAdam/v/transformer_denoiser/multi_head_attention/attention_output/kernelFAdam/m/transformer_denoiser/multi_head_attention/attention_output/biasFAdam/v/transformer_denoiser/multi_head_attention/attention_output/bias*Adam/m/transformer_denoiser/dense_3/kernel*Adam/v/transformer_denoiser/dense_3/kernel(Adam/m/transformer_denoiser/dense_3/bias(Adam/v/transformer_denoiser/dense_3/bias*Adam/m/transformer_denoiser/dense_4/kernel*Adam/v/transformer_denoiser/dense_4/kernel(Adam/m/transformer_denoiser/dense_4/bias(Adam/v/transformer_denoiser/dense_4/bias5Adam/m/transformer_denoiser/layer_normalization/gamma5Adam/v/transformer_denoiser/layer_normalization/gamma4Adam/m/transformer_denoiser/layer_normalization/beta4Adam/v/transformer_denoiser/layer_normalization/beta7Adam/m/transformer_denoiser/layer_normalization_1/gamma7Adam/v/transformer_denoiser/layer_normalization_1/gamma6Adam/m/transformer_denoiser/layer_normalization_1/beta6Adam/v/transformer_denoiser/layer_normalization_1/betatotalcountConst*S
TinL
J2H*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_218153
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!transformer_denoiser/dense/kerneltransformer_denoiser/dense/bias#transformer_denoiser/dense_1/kernel!transformer_denoiser/dense_1/bias#transformer_denoiser/dense_2/kernel!transformer_denoiser/dense_2/bias6transformer_denoiser/multi_head_attention/query/kernel4transformer_denoiser/multi_head_attention/query/bias4transformer_denoiser/multi_head_attention/key/kernel2transformer_denoiser/multi_head_attention/key/bias6transformer_denoiser/multi_head_attention/value/kernel4transformer_denoiser/multi_head_attention/value/biasAtransformer_denoiser/multi_head_attention/attention_output/kernel?transformer_denoiser/multi_head_attention/attention_output/bias#transformer_denoiser/dense_3/kernel!transformer_denoiser/dense_3/bias#transformer_denoiser/dense_4/kernel!transformer_denoiser/dense_4/bias.transformer_denoiser/layer_normalization/gamma-transformer_denoiser/layer_normalization/beta0transformer_denoiser/layer_normalization_1/gamma/transformer_denoiser/layer_normalization_1/beta	iterationlearning_rate(Adam/m/transformer_denoiser/dense/kernel(Adam/v/transformer_denoiser/dense/kernel&Adam/m/transformer_denoiser/dense/bias&Adam/v/transformer_denoiser/dense/bias*Adam/m/transformer_denoiser/dense_1/kernel*Adam/v/transformer_denoiser/dense_1/kernel(Adam/m/transformer_denoiser/dense_1/bias(Adam/v/transformer_denoiser/dense_1/bias*Adam/m/transformer_denoiser/dense_2/kernel*Adam/v/transformer_denoiser/dense_2/kernel(Adam/m/transformer_denoiser/dense_2/bias(Adam/v/transformer_denoiser/dense_2/bias=Adam/m/transformer_denoiser/multi_head_attention/query/kernel=Adam/v/transformer_denoiser/multi_head_attention/query/kernel;Adam/m/transformer_denoiser/multi_head_attention/query/bias;Adam/v/transformer_denoiser/multi_head_attention/query/bias;Adam/m/transformer_denoiser/multi_head_attention/key/kernel;Adam/v/transformer_denoiser/multi_head_attention/key/kernel9Adam/m/transformer_denoiser/multi_head_attention/key/bias9Adam/v/transformer_denoiser/multi_head_attention/key/bias=Adam/m/transformer_denoiser/multi_head_attention/value/kernel=Adam/v/transformer_denoiser/multi_head_attention/value/kernel;Adam/m/transformer_denoiser/multi_head_attention/value/bias;Adam/v/transformer_denoiser/multi_head_attention/value/biasHAdam/m/transformer_denoiser/multi_head_attention/attention_output/kernelHAdam/v/transformer_denoiser/multi_head_attention/attention_output/kernelFAdam/m/transformer_denoiser/multi_head_attention/attention_output/biasFAdam/v/transformer_denoiser/multi_head_attention/attention_output/bias*Adam/m/transformer_denoiser/dense_3/kernel*Adam/v/transformer_denoiser/dense_3/kernel(Adam/m/transformer_denoiser/dense_3/bias(Adam/v/transformer_denoiser/dense_3/bias*Adam/m/transformer_denoiser/dense_4/kernel*Adam/v/transformer_denoiser/dense_4/kernel(Adam/m/transformer_denoiser/dense_4/bias(Adam/v/transformer_denoiser/dense_4/bias5Adam/m/transformer_denoiser/layer_normalization/gamma5Adam/v/transformer_denoiser/layer_normalization/gamma4Adam/m/transformer_denoiser/layer_normalization/beta4Adam/v/transformer_denoiser/layer_normalization/beta7Adam/m/transformer_denoiser/layer_normalization_1/gamma7Adam/v/transformer_denoiser/layer_normalization_1/gamma6Adam/m/transformer_denoiser/layer_normalization_1/beta6Adam/v/transformer_denoiser/layer_normalization_1/betatotalcount*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_218372�
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_216987

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_217711

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
*__inference_dropout_1_layer_call_fn_217689

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_216951p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�-
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_217513	
query	
value
keyB
+query_einsum_einsum_readvariableop_resource:�@3
!query_add_readvariableop_resource:@@
)key_einsum_einsum_readvariableop_resource:�@1
key_add_readvariableop_resource:@B
+value_einsum_einsum_readvariableop_resource:�@3
!value_add_readvariableop_resource:@M
6attention_output_einsum_einsum_readvariableop_resource:@�;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
,
_output_shapes
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery
�
�
4__inference_layer_normalization_layer_call_fn_217604

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_216753p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name217600:&"
 
_user_specified_name217598:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_transformer_denoiser_layer_call_fn_217128
input_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	� 
	unknown_7:�@
	unknown_8:@ 
	unknown_9:�@

unknown_10:@!

unknown_11:�@

unknown_12:@!

unknown_13:@�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_transformer_denoiser_layer_call_and_return_conditional_losses_216970p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name217124:&"
 
_user_specified_name217122:&"
 
_user_specified_name217120:&"
 
_user_specified_name217118:&"
 
_user_specified_name217116:&"
 
_user_specified_name217114:&"
 
_user_specified_name217112:&"
 
_user_specified_name217110:&"
 
_user_specified_name217108:&"
 
_user_specified_name217106:&"
 
_user_specified_name217104:&"
 
_user_specified_name217102:&
"
 
_user_specified_name217100:&	"
 
_user_specified_name217098:&"
 
_user_specified_name217096:&"
 
_user_specified_name217094:&"
 
_user_specified_name217092:&"
 
_user_specified_name217090:&"
 
_user_specified_name217088:&"
 
_user_specified_name217086:&"
 
_user_specified_name217084:&"
 
_user_specified_name217082:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_217706

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�-
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_216870	
query	
value
keyB
+query_einsum_einsum_readvariableop_resource:�@3
!query_add_readvariableop_resource:@@
)key_einsum_einsum_readvariableop_resource:�@1
key_add_readvariableop_resource:@B
+value_einsum_einsum_readvariableop_resource:�@3
!value_add_readvariableop_resource:@M
6attention_output_einsum_einsum_readvariableop_resource:@�;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
,
_output_shapes
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_217431

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
(__inference_dropout_layer_call_fn_217662

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_216793p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_216934

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:����������T
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:����������O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:����������`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:����������^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
D
(__inference_dropout_layer_call_fn_217667

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_216987a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_217412

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:����������T
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:����������O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:����������`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:����������^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_217385

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:����������T
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:����������O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:����������`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:����������^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_3_layer_call_fn_217558

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_216934p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name217554:&"
 
_user_specified_name217552:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�D
�

P__inference_transformer_denoiser_layer_call_and_return_conditional_losses_216970
input_1)
layer_normalization_216754:	�)
layer_normalization_216756:	� 
dense_216777:
��
dense_216779:	�"
dense_1_216813:
��
dense_1_216815:	�"
dense_2_216828:
��
dense_2_216830:	�2
multi_head_attention_216871:�@-
multi_head_attention_216873:@2
multi_head_attention_216875:�@-
multi_head_attention_216877:@2
multi_head_attention_216879:�@-
multi_head_attention_216881:@2
multi_head_attention_216883:@�*
multi_head_attention_216885:	�+
layer_normalization_1_216912:	�+
layer_normalization_1_216914:	�"
dense_3_216935:
��
dense_3_216937:	�"
dense_4_216963:
��
dense_4_216965:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_216754layer_normalization_216756*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_216753�
dense/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_216777dense_216779*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_216776�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_216793�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_216813dense_1_216815*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_216812�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_216828dense_2_216830*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_216827P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�

ExpandDims
ExpandDims(dense_2/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:�����������
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0ExpandDims:output:0ExpandDims:output:0multi_head_attention_216871multi_head_attention_216873multi_head_attention_216875multi_head_attention_216877multi_head_attention_216879multi_head_attention_216881multi_head_attention_216883multi_head_attention_216885*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_216870�
SqueezeSqueeze5multi_head_attention/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������*
squeeze_dims
�
addAddV24layer_normalization/StatefulPartitionedCall:output:0Squeeze:output:0*
T0*(
_output_shapes
:�����������
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd:z:0layer_normalization_1_216912layer_normalization_1_216914*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_216911�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_3_216935dense_3_216937*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_216934�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_216951�
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_4_216963dense_4_216965*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_216962�
add_1AddV24layer_normalization/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentity	add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������: : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall:&"
 
_user_specified_name216965:&"
 
_user_specified_name216963:&"
 
_user_specified_name216937:&"
 
_user_specified_name216935:&"
 
_user_specified_name216914:&"
 
_user_specified_name216912:&"
 
_user_specified_name216885:&"
 
_user_specified_name216883:&"
 
_user_specified_name216881:&"
 
_user_specified_name216879:&"
 
_user_specified_name216877:&"
 
_user_specified_name216875:&
"
 
_user_specified_name216873:&	"
 
_user_specified_name216871:&"
 
_user_specified_name216830:&"
 
_user_specified_name216828:&"
 
_user_specified_name216815:&"
 
_user_specified_name216813:&"
 
_user_specified_name216779:&"
 
_user_specified_name216777:&"
 
_user_specified_name216756:&"
 
_user_specified_name216754:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
&__inference_dense_layer_call_fn_217367

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_216776p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name217363:&"
 
_user_specified_name217361:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_transformer_denoiser_layer_call_fn_217177
input_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	� 
	unknown_7:�@
	unknown_8:@ 
	unknown_9:�@

unknown_10:@!

unknown_11:�@

unknown_12:@!

unknown_13:@�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_transformer_denoiser_layer_call_and_return_conditional_losses_217079p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name217173:&"
 
_user_specified_name217171:&"
 
_user_specified_name217169:&"
 
_user_specified_name217167:&"
 
_user_specified_name217165:&"
 
_user_specified_name217163:&"
 
_user_specified_name217161:&"
 
_user_specified_name217159:&"
 
_user_specified_name217157:&"
 
_user_specified_name217155:&"
 
_user_specified_name217153:&"
 
_user_specified_name217151:&
"
 
_user_specified_name217149:&	"
 
_user_specified_name217147:&"
 
_user_specified_name217145:&"
 
_user_specified_name217143:&"
 
_user_specified_name217141:&"
 
_user_specified_name217139:&"
 
_user_specified_name217137:&"
 
_user_specified_name217135:&"
 
_user_specified_name217133:&"
 
_user_specified_name217131:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
��
�;
"__inference__traced_restore_218372
file_prefixF
2assignvariableop_transformer_denoiser_dense_kernel:
��A
2assignvariableop_1_transformer_denoiser_dense_bias:	�J
6assignvariableop_2_transformer_denoiser_dense_1_kernel:
��C
4assignvariableop_3_transformer_denoiser_dense_1_bias:	�J
6assignvariableop_4_transformer_denoiser_dense_2_kernel:
��C
4assignvariableop_5_transformer_denoiser_dense_2_bias:	�`
Iassignvariableop_6_transformer_denoiser_multi_head_attention_query_kernel:�@Y
Gassignvariableop_7_transformer_denoiser_multi_head_attention_query_bias:@^
Gassignvariableop_8_transformer_denoiser_multi_head_attention_key_kernel:�@W
Eassignvariableop_9_transformer_denoiser_multi_head_attention_key_bias:@a
Jassignvariableop_10_transformer_denoiser_multi_head_attention_value_kernel:�@Z
Hassignvariableop_11_transformer_denoiser_multi_head_attention_value_bias:@l
Uassignvariableop_12_transformer_denoiser_multi_head_attention_attention_output_kernel:@�b
Sassignvariableop_13_transformer_denoiser_multi_head_attention_attention_output_bias:	�K
7assignvariableop_14_transformer_denoiser_dense_3_kernel:
��D
5assignvariableop_15_transformer_denoiser_dense_3_bias:	�K
7assignvariableop_16_transformer_denoiser_dense_4_kernel:
��D
5assignvariableop_17_transformer_denoiser_dense_4_bias:	�Q
Bassignvariableop_18_transformer_denoiser_layer_normalization_gamma:	�P
Aassignvariableop_19_transformer_denoiser_layer_normalization_beta:	�S
Dassignvariableop_20_transformer_denoiser_layer_normalization_1_gamma:	�R
Cassignvariableop_21_transformer_denoiser_layer_normalization_1_beta:	�'
assignvariableop_22_iteration:	 +
!assignvariableop_23_learning_rate: P
<assignvariableop_24_adam_m_transformer_denoiser_dense_kernel:
��P
<assignvariableop_25_adam_v_transformer_denoiser_dense_kernel:
��I
:assignvariableop_26_adam_m_transformer_denoiser_dense_bias:	�I
:assignvariableop_27_adam_v_transformer_denoiser_dense_bias:	�R
>assignvariableop_28_adam_m_transformer_denoiser_dense_1_kernel:
��R
>assignvariableop_29_adam_v_transformer_denoiser_dense_1_kernel:
��K
<assignvariableop_30_adam_m_transformer_denoiser_dense_1_bias:	�K
<assignvariableop_31_adam_v_transformer_denoiser_dense_1_bias:	�R
>assignvariableop_32_adam_m_transformer_denoiser_dense_2_kernel:
��R
>assignvariableop_33_adam_v_transformer_denoiser_dense_2_kernel:
��K
<assignvariableop_34_adam_m_transformer_denoiser_dense_2_bias:	�K
<assignvariableop_35_adam_v_transformer_denoiser_dense_2_bias:	�h
Qassignvariableop_36_adam_m_transformer_denoiser_multi_head_attention_query_kernel:�@h
Qassignvariableop_37_adam_v_transformer_denoiser_multi_head_attention_query_kernel:�@a
Oassignvariableop_38_adam_m_transformer_denoiser_multi_head_attention_query_bias:@a
Oassignvariableop_39_adam_v_transformer_denoiser_multi_head_attention_query_bias:@f
Oassignvariableop_40_adam_m_transformer_denoiser_multi_head_attention_key_kernel:�@f
Oassignvariableop_41_adam_v_transformer_denoiser_multi_head_attention_key_kernel:�@_
Massignvariableop_42_adam_m_transformer_denoiser_multi_head_attention_key_bias:@_
Massignvariableop_43_adam_v_transformer_denoiser_multi_head_attention_key_bias:@h
Qassignvariableop_44_adam_m_transformer_denoiser_multi_head_attention_value_kernel:�@h
Qassignvariableop_45_adam_v_transformer_denoiser_multi_head_attention_value_kernel:�@a
Oassignvariableop_46_adam_m_transformer_denoiser_multi_head_attention_value_bias:@a
Oassignvariableop_47_adam_v_transformer_denoiser_multi_head_attention_value_bias:@s
\assignvariableop_48_adam_m_transformer_denoiser_multi_head_attention_attention_output_kernel:@�s
\assignvariableop_49_adam_v_transformer_denoiser_multi_head_attention_attention_output_kernel:@�i
Zassignvariableop_50_adam_m_transformer_denoiser_multi_head_attention_attention_output_bias:	�i
Zassignvariableop_51_adam_v_transformer_denoiser_multi_head_attention_attention_output_bias:	�R
>assignvariableop_52_adam_m_transformer_denoiser_dense_3_kernel:
��R
>assignvariableop_53_adam_v_transformer_denoiser_dense_3_kernel:
��K
<assignvariableop_54_adam_m_transformer_denoiser_dense_3_bias:	�K
<assignvariableop_55_adam_v_transformer_denoiser_dense_3_bias:	�R
>assignvariableop_56_adam_m_transformer_denoiser_dense_4_kernel:
��R
>assignvariableop_57_adam_v_transformer_denoiser_dense_4_kernel:
��K
<assignvariableop_58_adam_m_transformer_denoiser_dense_4_bias:	�K
<assignvariableop_59_adam_v_transformer_denoiser_dense_4_bias:	�X
Iassignvariableop_60_adam_m_transformer_denoiser_layer_normalization_gamma:	�X
Iassignvariableop_61_adam_v_transformer_denoiser_layer_normalization_gamma:	�W
Hassignvariableop_62_adam_m_transformer_denoiser_layer_normalization_beta:	�W
Hassignvariableop_63_adam_v_transformer_denoiser_layer_normalization_beta:	�Z
Kassignvariableop_64_adam_m_transformer_denoiser_layer_normalization_1_gamma:	�Z
Kassignvariableop_65_adam_v_transformer_denoiser_layer_normalization_1_gamma:	�Y
Jassignvariableop_66_adam_m_transformer_denoiser_layer_normalization_1_beta:	�Y
Jassignvariableop_67_adam_v_transformer_denoiser_layer_normalization_1_beta:	�#
assignvariableop_68_total: #
assignvariableop_69_count: 
identity_71��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*U
dtypesK
I2G	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp2assignvariableop_transformer_denoiser_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp2assignvariableop_1_transformer_denoiser_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp6assignvariableop_2_transformer_denoiser_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp4assignvariableop_3_transformer_denoiser_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp6assignvariableop_4_transformer_denoiser_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp4assignvariableop_5_transformer_denoiser_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpIassignvariableop_6_transformer_denoiser_multi_head_attention_query_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpGassignvariableop_7_transformer_denoiser_multi_head_attention_query_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpGassignvariableop_8_transformer_denoiser_multi_head_attention_key_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpEassignvariableop_9_transformer_denoiser_multi_head_attention_key_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpJassignvariableop_10_transformer_denoiser_multi_head_attention_value_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpHassignvariableop_11_transformer_denoiser_multi_head_attention_value_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpUassignvariableop_12_transformer_denoiser_multi_head_attention_attention_output_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpSassignvariableop_13_transformer_denoiser_multi_head_attention_attention_output_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp7assignvariableop_14_transformer_denoiser_dense_3_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp5assignvariableop_15_transformer_denoiser_dense_3_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp7assignvariableop_16_transformer_denoiser_dense_4_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp5assignvariableop_17_transformer_denoiser_dense_4_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpBassignvariableop_18_transformer_denoiser_layer_normalization_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpAassignvariableop_19_transformer_denoiser_layer_normalization_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpDassignvariableop_20_transformer_denoiser_layer_normalization_1_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpCassignvariableop_21_transformer_denoiser_layer_normalization_1_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_iterationIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_learning_rateIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp<assignvariableop_24_adam_m_transformer_denoiser_dense_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp<assignvariableop_25_adam_v_transformer_denoiser_dense_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp:assignvariableop_26_adam_m_transformer_denoiser_dense_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp:assignvariableop_27_adam_v_transformer_denoiser_dense_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_m_transformer_denoiser_dense_1_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_v_transformer_denoiser_dense_1_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp<assignvariableop_30_adam_m_transformer_denoiser_dense_1_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp<assignvariableop_31_adam_v_transformer_denoiser_dense_1_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp>assignvariableop_32_adam_m_transformer_denoiser_dense_2_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp>assignvariableop_33_adam_v_transformer_denoiser_dense_2_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp<assignvariableop_34_adam_m_transformer_denoiser_dense_2_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp<assignvariableop_35_adam_v_transformer_denoiser_dense_2_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpQassignvariableop_36_adam_m_transformer_denoiser_multi_head_attention_query_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpQassignvariableop_37_adam_v_transformer_denoiser_multi_head_attention_query_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpOassignvariableop_38_adam_m_transformer_denoiser_multi_head_attention_query_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpOassignvariableop_39_adam_v_transformer_denoiser_multi_head_attention_query_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpOassignvariableop_40_adam_m_transformer_denoiser_multi_head_attention_key_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpOassignvariableop_41_adam_v_transformer_denoiser_multi_head_attention_key_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpMassignvariableop_42_adam_m_transformer_denoiser_multi_head_attention_key_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpMassignvariableop_43_adam_v_transformer_denoiser_multi_head_attention_key_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpQassignvariableop_44_adam_m_transformer_denoiser_multi_head_attention_value_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpQassignvariableop_45_adam_v_transformer_denoiser_multi_head_attention_value_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpOassignvariableop_46_adam_m_transformer_denoiser_multi_head_attention_value_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpOassignvariableop_47_adam_v_transformer_denoiser_multi_head_attention_value_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp\assignvariableop_48_adam_m_transformer_denoiser_multi_head_attention_attention_output_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp\assignvariableop_49_adam_v_transformer_denoiser_multi_head_attention_attention_output_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpZassignvariableop_50_adam_m_transformer_denoiser_multi_head_attention_attention_output_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpZassignvariableop_51_adam_v_transformer_denoiser_multi_head_attention_attention_output_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp>assignvariableop_52_adam_m_transformer_denoiser_dense_3_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp>assignvariableop_53_adam_v_transformer_denoiser_dense_3_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp<assignvariableop_54_adam_m_transformer_denoiser_dense_3_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp<assignvariableop_55_adam_v_transformer_denoiser_dense_3_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp>assignvariableop_56_adam_m_transformer_denoiser_dense_4_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp>assignvariableop_57_adam_v_transformer_denoiser_dense_4_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp<assignvariableop_58_adam_m_transformer_denoiser_dense_4_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp<assignvariableop_59_adam_v_transformer_denoiser_dense_4_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpIassignvariableop_60_adam_m_transformer_denoiser_layer_normalization_gammaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpIassignvariableop_61_adam_v_transformer_denoiser_layer_normalization_gammaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpHassignvariableop_62_adam_m_transformer_denoiser_layer_normalization_betaIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpHassignvariableop_63_adam_v_transformer_denoiser_layer_normalization_betaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOpKassignvariableop_64_adam_m_transformer_denoiser_layer_normalization_1_gammaIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOpKassignvariableop_65_adam_v_transformer_denoiser_layer_normalization_1_gammaIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOpJassignvariableop_66_adam_m_transformer_denoiser_layer_normalization_1_betaIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOpJassignvariableop_67_adam_v_transformer_denoiser_layer_normalization_1_betaIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpassignvariableop_68_totalIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOpassignvariableop_69_countIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_70Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_71IdentityIdentity_70:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_71Identity_71:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%F!

_user_specified_namecount:%E!

_user_specified_nametotal:VDR
P
_user_specified_name86Adam/v/transformer_denoiser/layer_normalization_1/beta:VCR
P
_user_specified_name86Adam/m/transformer_denoiser/layer_normalization_1/beta:WBS
Q
_user_specified_name97Adam/v/transformer_denoiser/layer_normalization_1/gamma:WAS
Q
_user_specified_name97Adam/m/transformer_denoiser/layer_normalization_1/gamma:T@P
N
_user_specified_name64Adam/v/transformer_denoiser/layer_normalization/beta:T?P
N
_user_specified_name64Adam/m/transformer_denoiser/layer_normalization/beta:U>Q
O
_user_specified_name75Adam/v/transformer_denoiser/layer_normalization/gamma:U=Q
O
_user_specified_name75Adam/m/transformer_denoiser/layer_normalization/gamma:H<D
B
_user_specified_name*(Adam/v/transformer_denoiser/dense_4/bias:H;D
B
_user_specified_name*(Adam/m/transformer_denoiser/dense_4/bias:J:F
D
_user_specified_name,*Adam/v/transformer_denoiser/dense_4/kernel:J9F
D
_user_specified_name,*Adam/m/transformer_denoiser/dense_4/kernel:H8D
B
_user_specified_name*(Adam/v/transformer_denoiser/dense_3/bias:H7D
B
_user_specified_name*(Adam/m/transformer_denoiser/dense_3/bias:J6F
D
_user_specified_name,*Adam/v/transformer_denoiser/dense_3/kernel:J5F
D
_user_specified_name,*Adam/m/transformer_denoiser/dense_3/kernel:f4b
`
_user_specified_nameHFAdam/v/transformer_denoiser/multi_head_attention/attention_output/bias:f3b
`
_user_specified_nameHFAdam/m/transformer_denoiser/multi_head_attention/attention_output/bias:h2d
b
_user_specified_nameJHAdam/v/transformer_denoiser/multi_head_attention/attention_output/kernel:h1d
b
_user_specified_nameJHAdam/m/transformer_denoiser/multi_head_attention/attention_output/kernel:[0W
U
_user_specified_name=;Adam/v/transformer_denoiser/multi_head_attention/value/bias:[/W
U
_user_specified_name=;Adam/m/transformer_denoiser/multi_head_attention/value/bias:].Y
W
_user_specified_name?=Adam/v/transformer_denoiser/multi_head_attention/value/kernel:]-Y
W
_user_specified_name?=Adam/m/transformer_denoiser/multi_head_attention/value/kernel:Y,U
S
_user_specified_name;9Adam/v/transformer_denoiser/multi_head_attention/key/bias:Y+U
S
_user_specified_name;9Adam/m/transformer_denoiser/multi_head_attention/key/bias:[*W
U
_user_specified_name=;Adam/v/transformer_denoiser/multi_head_attention/key/kernel:[)W
U
_user_specified_name=;Adam/m/transformer_denoiser/multi_head_attention/key/kernel:[(W
U
_user_specified_name=;Adam/v/transformer_denoiser/multi_head_attention/query/bias:['W
U
_user_specified_name=;Adam/m/transformer_denoiser/multi_head_attention/query/bias:]&Y
W
_user_specified_name?=Adam/v/transformer_denoiser/multi_head_attention/query/kernel:]%Y
W
_user_specified_name?=Adam/m/transformer_denoiser/multi_head_attention/query/kernel:H$D
B
_user_specified_name*(Adam/v/transformer_denoiser/dense_2/bias:H#D
B
_user_specified_name*(Adam/m/transformer_denoiser/dense_2/bias:J"F
D
_user_specified_name,*Adam/v/transformer_denoiser/dense_2/kernel:J!F
D
_user_specified_name,*Adam/m/transformer_denoiser/dense_2/kernel:H D
B
_user_specified_name*(Adam/v/transformer_denoiser/dense_1/bias:HD
B
_user_specified_name*(Adam/m/transformer_denoiser/dense_1/bias:JF
D
_user_specified_name,*Adam/v/transformer_denoiser/dense_1/kernel:JF
D
_user_specified_name,*Adam/m/transformer_denoiser/dense_1/kernel:FB
@
_user_specified_name(&Adam/v/transformer_denoiser/dense/bias:FB
@
_user_specified_name(&Adam/m/transformer_denoiser/dense/bias:HD
B
_user_specified_name*(Adam/v/transformer_denoiser/dense/kernel:HD
B
_user_specified_name*(Adam/m/transformer_denoiser/dense/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:OK
I
_user_specified_name1/transformer_denoiser/layer_normalization_1/beta:PL
J
_user_specified_name20transformer_denoiser/layer_normalization_1/gamma:MI
G
_user_specified_name/-transformer_denoiser/layer_normalization/beta:NJ
H
_user_specified_name0.transformer_denoiser/layer_normalization/gamma:A=
;
_user_specified_name#!transformer_denoiser/dense_4/bias:C?
=
_user_specified_name%#transformer_denoiser/dense_4/kernel:A=
;
_user_specified_name#!transformer_denoiser/dense_3/bias:C?
=
_user_specified_name%#transformer_denoiser/dense_3/kernel:_[
Y
_user_specified_nameA?transformer_denoiser/multi_head_attention/attention_output/bias:a]
[
_user_specified_nameCAtransformer_denoiser/multi_head_attention/attention_output/kernel:TP
N
_user_specified_name64transformer_denoiser/multi_head_attention/value/bias:VR
P
_user_specified_name86transformer_denoiser/multi_head_attention/value/kernel:R
N
L
_user_specified_name42transformer_denoiser/multi_head_attention/key/bias:T	P
N
_user_specified_name64transformer_denoiser/multi_head_attention/key/kernel:TP
N
_user_specified_name64transformer_denoiser/multi_head_attention/query/bias:VR
P
_user_specified_name86transformer_denoiser/multi_head_attention/query/kernel:A=
;
_user_specified_name#!transformer_denoiser/dense_2/bias:C?
=
_user_specified_name%#transformer_denoiser/dense_2/kernel:A=
;
_user_specified_name#!transformer_denoiser/dense_1/bias:C?
=
_user_specified_name%#transformer_denoiser/dense_1/kernel:?;
9
_user_specified_name!transformer_denoiser/dense/bias:A=
;
_user_specified_name#!transformer_denoiser/dense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_216753

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5}
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*'
_output_shapes
:���������]
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*'
_output_shapes
:���������
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������s
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*(
_output_shapes
:����������w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0~
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*(
_output_shapes
:����������s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_4_layer_call_and_return_conditional_losses_217595

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_216776

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:����������T
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:����������O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:����������`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:����������^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_216812

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:����������T
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:����������O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:����������`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:����������^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�-
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_217549	
query	
value
keyB
+query_einsum_einsum_readvariableop_resource:�@3
!query_add_readvariableop_resource:@@
)key_einsum_einsum_readvariableop_resource:�@1
key_add_readvariableop_resource:@B
+value_einsum_einsum_readvariableop_resource:�@3
!value_add_readvariableop_resource:@M
6attention_output_einsum_einsum_readvariableop_resource:@�;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
,
_output_shapes
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery
�
�
(__inference_dense_2_layer_call_fn_217421

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_216827p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name217417:&"
 
_user_specified_name217415:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_216729
input_1]
Ntransformer_denoiser_layer_normalization_batchnorm_mul_readvariableop_resource:	�Y
Jtransformer_denoiser_layer_normalization_batchnorm_readvariableop_resource:	�M
9transformer_denoiser_dense_matmul_readvariableop_resource:
��I
:transformer_denoiser_dense_biasadd_readvariableop_resource:	�O
;transformer_denoiser_dense_1_matmul_readvariableop_resource:
��K
<transformer_denoiser_dense_1_biasadd_readvariableop_resource:	�O
;transformer_denoiser_dense_2_matmul_readvariableop_resource:
��K
<transformer_denoiser_dense_2_biasadd_readvariableop_resource:	�l
Utransformer_denoiser_multi_head_attention_query_einsum_einsum_readvariableop_resource:�@]
Ktransformer_denoiser_multi_head_attention_query_add_readvariableop_resource:@j
Stransformer_denoiser_multi_head_attention_key_einsum_einsum_readvariableop_resource:�@[
Itransformer_denoiser_multi_head_attention_key_add_readvariableop_resource:@l
Utransformer_denoiser_multi_head_attention_value_einsum_einsum_readvariableop_resource:�@]
Ktransformer_denoiser_multi_head_attention_value_add_readvariableop_resource:@w
`transformer_denoiser_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:@�e
Vtransformer_denoiser_multi_head_attention_attention_output_add_readvariableop_resource:	�_
Ptransformer_denoiser_layer_normalization_1_batchnorm_mul_readvariableop_resource:	�[
Ltransformer_denoiser_layer_normalization_1_batchnorm_readvariableop_resource:	�O
;transformer_denoiser_dense_3_matmul_readvariableop_resource:
��K
<transformer_denoiser_dense_3_biasadd_readvariableop_resource:	�O
;transformer_denoiser_dense_4_matmul_readvariableop_resource:
��K
<transformer_denoiser_dense_4_biasadd_readvariableop_resource:	�
identity��1transformer_denoiser/dense/BiasAdd/ReadVariableOp�0transformer_denoiser/dense/MatMul/ReadVariableOp�3transformer_denoiser/dense_1/BiasAdd/ReadVariableOp�2transformer_denoiser/dense_1/MatMul/ReadVariableOp�3transformer_denoiser/dense_2/BiasAdd/ReadVariableOp�2transformer_denoiser/dense_2/MatMul/ReadVariableOp�3transformer_denoiser/dense_3/BiasAdd/ReadVariableOp�2transformer_denoiser/dense_3/MatMul/ReadVariableOp�3transformer_denoiser/dense_4/BiasAdd/ReadVariableOp�2transformer_denoiser/dense_4/MatMul/ReadVariableOp�Atransformer_denoiser/layer_normalization/batchnorm/ReadVariableOp�Etransformer_denoiser/layer_normalization/batchnorm/mul/ReadVariableOp�Ctransformer_denoiser/layer_normalization_1/batchnorm/ReadVariableOp�Gtransformer_denoiser/layer_normalization_1/batchnorm/mul/ReadVariableOp�Mtransformer_denoiser/multi_head_attention/attention_output/add/ReadVariableOp�Wtransformer_denoiser/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�@transformer_denoiser/multi_head_attention/key/add/ReadVariableOp�Jtransformer_denoiser/multi_head_attention/key/einsum/Einsum/ReadVariableOp�Btransformer_denoiser/multi_head_attention/query/add/ReadVariableOp�Ltransformer_denoiser/multi_head_attention/query/einsum/Einsum/ReadVariableOp�Btransformer_denoiser/multi_head_attention/value/add/ReadVariableOp�Ltransformer_denoiser/multi_head_attention/value/einsum/Einsum/ReadVariableOp�
Gtransformer_denoiser/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
5transformer_denoiser/layer_normalization/moments/meanMeaninput_1Ptransformer_denoiser/layer_normalization/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(�
=transformer_denoiser/layer_normalization/moments/StopGradientStopGradient>transformer_denoiser/layer_normalization/moments/mean:output:0*
T0*'
_output_shapes
:����������
Btransformer_denoiser/layer_normalization/moments/SquaredDifferenceSquaredDifferenceinput_1Ftransformer_denoiser/layer_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Ktransformer_denoiser/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
9transformer_denoiser/layer_normalization/moments/varianceMeanFtransformer_denoiser/layer_normalization/moments/SquaredDifference:z:0Ttransformer_denoiser/layer_normalization/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(}
8transformer_denoiser/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
6transformer_denoiser/layer_normalization/batchnorm/addAddV2Btransformer_denoiser/layer_normalization/moments/variance:output:0Atransformer_denoiser/layer_normalization/batchnorm/add/y:output:0*
T0*'
_output_shapes
:����������
8transformer_denoiser/layer_normalization/batchnorm/RsqrtRsqrt:transformer_denoiser/layer_normalization/batchnorm/add:z:0*
T0*'
_output_shapes
:����������
Etransformer_denoiser/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpNtransformer_denoiser_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6transformer_denoiser/layer_normalization/batchnorm/mulMul<transformer_denoiser/layer_normalization/batchnorm/Rsqrt:y:0Mtransformer_denoiser/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8transformer_denoiser/layer_normalization/batchnorm/mul_1Mulinput_1:transformer_denoiser/layer_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
8transformer_denoiser/layer_normalization/batchnorm/mul_2Mul>transformer_denoiser/layer_normalization/moments/mean:output:0:transformer_denoiser/layer_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Atransformer_denoiser/layer_normalization/batchnorm/ReadVariableOpReadVariableOpJtransformer_denoiser_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6transformer_denoiser/layer_normalization/batchnorm/subSubItransformer_denoiser/layer_normalization/batchnorm/ReadVariableOp:value:0<transformer_denoiser/layer_normalization/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:�����������
8transformer_denoiser/layer_normalization/batchnorm/add_1AddV2<transformer_denoiser/layer_normalization/batchnorm/mul_1:z:0:transformer_denoiser/layer_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
0transformer_denoiser/dense/MatMul/ReadVariableOpReadVariableOp9transformer_denoiser_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
!transformer_denoiser/dense/MatMulMatMul<transformer_denoiser/layer_normalization/batchnorm/add_1:z:08transformer_denoiser/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1transformer_denoiser/dense/BiasAdd/ReadVariableOpReadVariableOp:transformer_denoiser_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"transformer_denoiser/dense/BiasAddBiasAdd+transformer_denoiser/dense/MatMul:product:09transformer_denoiser/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������j
%transformer_denoiser/dense/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
#transformer_denoiser/dense/Gelu/mulMul.transformer_denoiser/dense/Gelu/mul/x:output:0+transformer_denoiser/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������k
&transformer_denoiser/dense/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
'transformer_denoiser/dense/Gelu/truedivRealDiv+transformer_denoiser/dense/BiasAdd:output:0/transformer_denoiser/dense/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:�����������
#transformer_denoiser/dense/Gelu/ErfErf+transformer_denoiser/dense/Gelu/truediv:z:0*
T0*(
_output_shapes
:����������j
%transformer_denoiser/dense/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#transformer_denoiser/dense/Gelu/addAddV2.transformer_denoiser/dense/Gelu/add/x:output:0'transformer_denoiser/dense/Gelu/Erf:y:0*
T0*(
_output_shapes
:�����������
%transformer_denoiser/dense/Gelu/mul_1Mul'transformer_denoiser/dense/Gelu/mul:z:0'transformer_denoiser/dense/Gelu/add:z:0*
T0*(
_output_shapes
:�����������
%transformer_denoiser/dropout/IdentityIdentity)transformer_denoiser/dense/Gelu/mul_1:z:0*
T0*(
_output_shapes
:�����������
2transformer_denoiser/dense_1/MatMul/ReadVariableOpReadVariableOp;transformer_denoiser_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
#transformer_denoiser/dense_1/MatMulMatMul.transformer_denoiser/dropout/Identity:output:0:transformer_denoiser/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
3transformer_denoiser/dense_1/BiasAdd/ReadVariableOpReadVariableOp<transformer_denoiser_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$transformer_denoiser/dense_1/BiasAddBiasAdd-transformer_denoiser/dense_1/MatMul:product:0;transformer_denoiser/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
'transformer_denoiser/dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%transformer_denoiser/dense_1/Gelu/mulMul0transformer_denoiser/dense_1/Gelu/mul/x:output:0-transformer_denoiser/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������m
(transformer_denoiser/dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)transformer_denoiser/dense_1/Gelu/truedivRealDiv-transformer_denoiser/dense_1/BiasAdd:output:01transformer_denoiser/dense_1/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:�����������
%transformer_denoiser/dense_1/Gelu/ErfErf-transformer_denoiser/dense_1/Gelu/truediv:z:0*
T0*(
_output_shapes
:����������l
'transformer_denoiser/dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%transformer_denoiser/dense_1/Gelu/addAddV20transformer_denoiser/dense_1/Gelu/add/x:output:0)transformer_denoiser/dense_1/Gelu/Erf:y:0*
T0*(
_output_shapes
:�����������
'transformer_denoiser/dense_1/Gelu/mul_1Mul)transformer_denoiser/dense_1/Gelu/mul:z:0)transformer_denoiser/dense_1/Gelu/add:z:0*
T0*(
_output_shapes
:�����������
2transformer_denoiser/dense_2/MatMul/ReadVariableOpReadVariableOp;transformer_denoiser_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
#transformer_denoiser/dense_2/MatMulMatMul+transformer_denoiser/dense_1/Gelu/mul_1:z:0:transformer_denoiser/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
3transformer_denoiser/dense_2/BiasAdd/ReadVariableOpReadVariableOp<transformer_denoiser_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$transformer_denoiser/dense_2/BiasAddBiasAdd-transformer_denoiser/dense_2/MatMul:product:0;transformer_denoiser/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
#transformer_denoiser/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
transformer_denoiser/ExpandDims
ExpandDims-transformer_denoiser/dense_2/BiasAdd:output:0,transformer_denoiser/ExpandDims/dim:output:0*
T0*,
_output_shapes
:�����������
Ltransformer_denoiser/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpUtransformer_denoiser_multi_head_attention_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
=transformer_denoiser/multi_head_attention/query/einsum/EinsumEinsum(transformer_denoiser/ExpandDims:output:0Ttransformer_denoiser/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
Btransformer_denoiser/multi_head_attention/query/add/ReadVariableOpReadVariableOpKtransformer_denoiser_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
3transformer_denoiser/multi_head_attention/query/addAddV2Ftransformer_denoiser/multi_head_attention/query/einsum/Einsum:output:0Jtransformer_denoiser/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
Jtransformer_denoiser/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpStransformer_denoiser_multi_head_attention_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
;transformer_denoiser/multi_head_attention/key/einsum/EinsumEinsum(transformer_denoiser/ExpandDims:output:0Rtransformer_denoiser/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
@transformer_denoiser/multi_head_attention/key/add/ReadVariableOpReadVariableOpItransformer_denoiser_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
1transformer_denoiser/multi_head_attention/key/addAddV2Dtransformer_denoiser/multi_head_attention/key/einsum/Einsum:output:0Htransformer_denoiser/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
Ltransformer_denoiser/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpUtransformer_denoiser_multi_head_attention_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
=transformer_denoiser/multi_head_attention/value/einsum/EinsumEinsum(transformer_denoiser/ExpandDims:output:0Ttransformer_denoiser/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
Btransformer_denoiser/multi_head_attention/value/add/ReadVariableOpReadVariableOpKtransformer_denoiser_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
3transformer_denoiser/multi_head_attention/value/addAddV2Ftransformer_denoiser/multi_head_attention/value/einsum/Einsum:output:0Jtransformer_denoiser/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@t
/transformer_denoiser/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
-transformer_denoiser/multi_head_attention/MulMul7transformer_denoiser/multi_head_attention/query/add:z:08transformer_denoiser/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:���������@�
7transformer_denoiser/multi_head_attention/einsum/EinsumEinsum5transformer_denoiser/multi_head_attention/key/add:z:01transformer_denoiser/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
9transformer_denoiser/multi_head_attention/softmax/SoftmaxSoftmax@transformer_denoiser/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
:transformer_denoiser/multi_head_attention/dropout/IdentityIdentityCtransformer_denoiser/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
9transformer_denoiser/multi_head_attention/einsum_1/EinsumEinsumCtransformer_denoiser/multi_head_attention/dropout/Identity:output:07transformer_denoiser/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
Wtransformer_denoiser/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOp`transformer_denoiser_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
Htransformer_denoiser/multi_head_attention/attention_output/einsum/EinsumEinsumBtransformer_denoiser/multi_head_attention/einsum_1/Einsum:output:0_transformer_denoiser/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
Mtransformer_denoiser/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpVtransformer_denoiser_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
>transformer_denoiser/multi_head_attention/attention_output/addAddV2Qtransformer_denoiser/multi_head_attention/attention_output/einsum/Einsum:output:0Utransformer_denoiser/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
transformer_denoiser/SqueezeSqueezeBtransformer_denoiser/multi_head_attention/attention_output/add:z:0*
T0*(
_output_shapes
:����������*
squeeze_dims
�
transformer_denoiser/addAddV2<transformer_denoiser/layer_normalization/batchnorm/add_1:z:0%transformer_denoiser/Squeeze:output:0*
T0*(
_output_shapes
:�����������
Itransformer_denoiser/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
7transformer_denoiser/layer_normalization_1/moments/meanMeantransformer_denoiser/add:z:0Rtransformer_denoiser/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(�
?transformer_denoiser/layer_normalization_1/moments/StopGradientStopGradient@transformer_denoiser/layer_normalization_1/moments/mean:output:0*
T0*'
_output_shapes
:����������
Dtransformer_denoiser/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_denoiser/add:z:0Htransformer_denoiser/layer_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Mtransformer_denoiser/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
;transformer_denoiser/layer_normalization_1/moments/varianceMeanHtransformer_denoiser/layer_normalization_1/moments/SquaredDifference:z:0Vtransformer_denoiser/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(
:transformer_denoiser/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
8transformer_denoiser/layer_normalization_1/batchnorm/addAddV2Dtransformer_denoiser/layer_normalization_1/moments/variance:output:0Ctransformer_denoiser/layer_normalization_1/batchnorm/add/y:output:0*
T0*'
_output_shapes
:����������
:transformer_denoiser/layer_normalization_1/batchnorm/RsqrtRsqrt<transformer_denoiser/layer_normalization_1/batchnorm/add:z:0*
T0*'
_output_shapes
:����������
Gtransformer_denoiser/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_denoiser_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8transformer_denoiser/layer_normalization_1/batchnorm/mulMul>transformer_denoiser/layer_normalization_1/batchnorm/Rsqrt:y:0Otransformer_denoiser/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:transformer_denoiser/layer_normalization_1/batchnorm/mul_1Multransformer_denoiser/add:z:0<transformer_denoiser/layer_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
:transformer_denoiser/layer_normalization_1/batchnorm/mul_2Mul@transformer_denoiser/layer_normalization_1/moments/mean:output:0<transformer_denoiser/layer_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Ctransformer_denoiser/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpLtransformer_denoiser_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8transformer_denoiser/layer_normalization_1/batchnorm/subSubKtransformer_denoiser/layer_normalization_1/batchnorm/ReadVariableOp:value:0>transformer_denoiser/layer_normalization_1/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:�����������
:transformer_denoiser/layer_normalization_1/batchnorm/add_1AddV2>transformer_denoiser/layer_normalization_1/batchnorm/mul_1:z:0<transformer_denoiser/layer_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
2transformer_denoiser/dense_3/MatMul/ReadVariableOpReadVariableOp;transformer_denoiser_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
#transformer_denoiser/dense_3/MatMulMatMul>transformer_denoiser/layer_normalization_1/batchnorm/add_1:z:0:transformer_denoiser/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
3transformer_denoiser/dense_3/BiasAdd/ReadVariableOpReadVariableOp<transformer_denoiser_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$transformer_denoiser/dense_3/BiasAddBiasAdd-transformer_denoiser/dense_3/MatMul:product:0;transformer_denoiser/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
'transformer_denoiser/dense_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
%transformer_denoiser/dense_3/Gelu/mulMul0transformer_denoiser/dense_3/Gelu/mul/x:output:0-transformer_denoiser/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������m
(transformer_denoiser/dense_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
)transformer_denoiser/dense_3/Gelu/truedivRealDiv-transformer_denoiser/dense_3/BiasAdd:output:01transformer_denoiser/dense_3/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:�����������
%transformer_denoiser/dense_3/Gelu/ErfErf-transformer_denoiser/dense_3/Gelu/truediv:z:0*
T0*(
_output_shapes
:����������l
'transformer_denoiser/dense_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%transformer_denoiser/dense_3/Gelu/addAddV20transformer_denoiser/dense_3/Gelu/add/x:output:0)transformer_denoiser/dense_3/Gelu/Erf:y:0*
T0*(
_output_shapes
:�����������
'transformer_denoiser/dense_3/Gelu/mul_1Mul)transformer_denoiser/dense_3/Gelu/mul:z:0)transformer_denoiser/dense_3/Gelu/add:z:0*
T0*(
_output_shapes
:�����������
'transformer_denoiser/dropout_1/IdentityIdentity+transformer_denoiser/dense_3/Gelu/mul_1:z:0*
T0*(
_output_shapes
:�����������
2transformer_denoiser/dense_4/MatMul/ReadVariableOpReadVariableOp;transformer_denoiser_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
#transformer_denoiser/dense_4/MatMulMatMul0transformer_denoiser/dropout_1/Identity:output:0:transformer_denoiser/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
3transformer_denoiser/dense_4/BiasAdd/ReadVariableOpReadVariableOp<transformer_denoiser_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$transformer_denoiser/dense_4/BiasAddBiasAdd-transformer_denoiser/dense_4/MatMul:product:0;transformer_denoiser/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
transformer_denoiser/add_1AddV2<transformer_denoiser/layer_normalization/batchnorm/add_1:z:0-transformer_denoiser/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:����������n
IdentityIdentitytransformer_denoiser/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp2^transformer_denoiser/dense/BiasAdd/ReadVariableOp1^transformer_denoiser/dense/MatMul/ReadVariableOp4^transformer_denoiser/dense_1/BiasAdd/ReadVariableOp3^transformer_denoiser/dense_1/MatMul/ReadVariableOp4^transformer_denoiser/dense_2/BiasAdd/ReadVariableOp3^transformer_denoiser/dense_2/MatMul/ReadVariableOp4^transformer_denoiser/dense_3/BiasAdd/ReadVariableOp3^transformer_denoiser/dense_3/MatMul/ReadVariableOp4^transformer_denoiser/dense_4/BiasAdd/ReadVariableOp3^transformer_denoiser/dense_4/MatMul/ReadVariableOpB^transformer_denoiser/layer_normalization/batchnorm/ReadVariableOpF^transformer_denoiser/layer_normalization/batchnorm/mul/ReadVariableOpD^transformer_denoiser/layer_normalization_1/batchnorm/ReadVariableOpH^transformer_denoiser/layer_normalization_1/batchnorm/mul/ReadVariableOpN^transformer_denoiser/multi_head_attention/attention_output/add/ReadVariableOpX^transformer_denoiser/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpA^transformer_denoiser/multi_head_attention/key/add/ReadVariableOpK^transformer_denoiser/multi_head_attention/key/einsum/Einsum/ReadVariableOpC^transformer_denoiser/multi_head_attention/query/add/ReadVariableOpM^transformer_denoiser/multi_head_attention/query/einsum/Einsum/ReadVariableOpC^transformer_denoiser/multi_head_attention/value/add/ReadVariableOpM^transformer_denoiser/multi_head_attention/value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������: : : : : : : : : : : : : : : : : : : : : : 2f
1transformer_denoiser/dense/BiasAdd/ReadVariableOp1transformer_denoiser/dense/BiasAdd/ReadVariableOp2d
0transformer_denoiser/dense/MatMul/ReadVariableOp0transformer_denoiser/dense/MatMul/ReadVariableOp2j
3transformer_denoiser/dense_1/BiasAdd/ReadVariableOp3transformer_denoiser/dense_1/BiasAdd/ReadVariableOp2h
2transformer_denoiser/dense_1/MatMul/ReadVariableOp2transformer_denoiser/dense_1/MatMul/ReadVariableOp2j
3transformer_denoiser/dense_2/BiasAdd/ReadVariableOp3transformer_denoiser/dense_2/BiasAdd/ReadVariableOp2h
2transformer_denoiser/dense_2/MatMul/ReadVariableOp2transformer_denoiser/dense_2/MatMul/ReadVariableOp2j
3transformer_denoiser/dense_3/BiasAdd/ReadVariableOp3transformer_denoiser/dense_3/BiasAdd/ReadVariableOp2h
2transformer_denoiser/dense_3/MatMul/ReadVariableOp2transformer_denoiser/dense_3/MatMul/ReadVariableOp2j
3transformer_denoiser/dense_4/BiasAdd/ReadVariableOp3transformer_denoiser/dense_4/BiasAdd/ReadVariableOp2h
2transformer_denoiser/dense_4/MatMul/ReadVariableOp2transformer_denoiser/dense_4/MatMul/ReadVariableOp2�
Atransformer_denoiser/layer_normalization/batchnorm/ReadVariableOpAtransformer_denoiser/layer_normalization/batchnorm/ReadVariableOp2�
Etransformer_denoiser/layer_normalization/batchnorm/mul/ReadVariableOpEtransformer_denoiser/layer_normalization/batchnorm/mul/ReadVariableOp2�
Ctransformer_denoiser/layer_normalization_1/batchnorm/ReadVariableOpCtransformer_denoiser/layer_normalization_1/batchnorm/ReadVariableOp2�
Gtransformer_denoiser/layer_normalization_1/batchnorm/mul/ReadVariableOpGtransformer_denoiser/layer_normalization_1/batchnorm/mul/ReadVariableOp2�
Mtransformer_denoiser/multi_head_attention/attention_output/add/ReadVariableOpMtransformer_denoiser/multi_head_attention/attention_output/add/ReadVariableOp2�
Wtransformer_denoiser/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpWtransformer_denoiser/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2�
@transformer_denoiser/multi_head_attention/key/add/ReadVariableOp@transformer_denoiser/multi_head_attention/key/add/ReadVariableOp2�
Jtransformer_denoiser/multi_head_attention/key/einsum/Einsum/ReadVariableOpJtransformer_denoiser/multi_head_attention/key/einsum/Einsum/ReadVariableOp2�
Btransformer_denoiser/multi_head_attention/query/add/ReadVariableOpBtransformer_denoiser/multi_head_attention/query/add/ReadVariableOp2�
Ltransformer_denoiser/multi_head_attention/query/einsum/Einsum/ReadVariableOpLtransformer_denoiser/multi_head_attention/query/einsum/Einsum/ReadVariableOp2�
Btransformer_denoiser/multi_head_attention/value/add/ReadVariableOpBtransformer_denoiser/multi_head_attention/value/add/ReadVariableOp2�
Ltransformer_denoiser/multi_head_attention/value/einsum/Einsum/ReadVariableOpLtransformer_denoiser/multi_head_attention/value/einsum/Einsum/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_216793

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_1_layer_call_fn_217394

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_216812p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name217390:&"
 
_user_specified_name217388:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_216911

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5}
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*'
_output_shapes
:���������]
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*'
_output_shapes
:���������
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������s
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*(
_output_shapes
:����������w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0~
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*(
_output_shapes
:����������s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_217657

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5}
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*'
_output_shapes
:���������]
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*'
_output_shapes
:���������
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������s
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*(
_output_shapes
:����������w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0~
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*(
_output_shapes
:����������s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_multi_head_attention_layer_call_fn_217454	
query	
value
key
unknown:�@
	unknown_0:@ 
	unknown_1:�@
	unknown_2:@ 
	unknown_3:�@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_216870t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&
"
 
_user_specified_name217450:&	"
 
_user_specified_name217448:&"
 
_user_specified_name217446:&"
 
_user_specified_name217444:&"
 
_user_specified_name217442:&"
 
_user_specified_name217440:&"
 
_user_specified_name217438:&"
 
_user_specified_name217436:QM
,
_output_shapes
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery
�
F
*__inference_dropout_1_layer_call_fn_217694

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_217070a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_217070

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_217679

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_4_layer_call_fn_217585

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_216962p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name217581:&"
 
_user_specified_name217579:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_multi_head_attention_layer_call_fn_217477	
query	
value
key
unknown:�@
	unknown_0:@ 
	unknown_1:�@
	unknown_2:@ 
	unknown_3:�@
	unknown_4:@ 
	unknown_5:@�
	unknown_6:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_217036t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&
"
 
_user_specified_name217473:&	"
 
_user_specified_name217471:&"
 
_user_specified_name217469:&"
 
_user_specified_name217467:&"
 
_user_specified_name217465:&"
 
_user_specified_name217463:&"
 
_user_specified_name217461:&"
 
_user_specified_name217459:QM
,
_output_shapes
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery
�

�
C__inference_dense_4_layer_call_and_return_conditional_losses_216962

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_217684

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�A
�

P__inference_transformer_denoiser_layer_call_and_return_conditional_losses_217079
input_1)
layer_normalization_216973:	�)
layer_normalization_216975:	� 
dense_216978:
��
dense_216980:	�"
dense_1_216989:
��
dense_1_216991:	�"
dense_2_216994:
��
dense_2_216996:	�2
multi_head_attention_217037:�@-
multi_head_attention_217039:@2
multi_head_attention_217041:�@-
multi_head_attention_217043:@2
multi_head_attention_217045:�@-
multi_head_attention_217047:@2
multi_head_attention_217049:@�*
multi_head_attention_217051:	�+
layer_normalization_1_217056:	�+
layer_normalization_1_217058:	�"
dense_3_217061:
��
dense_3_217063:	�"
dense_4_217072:
��
dense_4_217074:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCallinput_1layer_normalization_216973layer_normalization_216975*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_216753�
dense/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_216978dense_216980*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_216776�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_216987�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_216989dense_1_216991*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_216812�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_216994dense_2_216996*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_216827P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�

ExpandDims
ExpandDims(dense_2/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:�����������
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0ExpandDims:output:0ExpandDims:output:0multi_head_attention_217037multi_head_attention_217039multi_head_attention_217041multi_head_attention_217043multi_head_attention_217045multi_head_attention_217047multi_head_attention_217049multi_head_attention_217051*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_217036�
SqueezeSqueeze5multi_head_attention/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������*
squeeze_dims
�
addAddV24layer_normalization/StatefulPartitionedCall:output:0Squeeze:output:0*
T0*(
_output_shapes
:�����������
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd:z:0layer_normalization_1_217056layer_normalization_1_217058*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_216911�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_3_217061dense_3_217063*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_216934�
dropout_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_217070�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_4_217072dense_4_217074*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_216962�
add_1AddV24layer_normalization/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentity	add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������: : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall:&"
 
_user_specified_name217074:&"
 
_user_specified_name217072:&"
 
_user_specified_name217063:&"
 
_user_specified_name217061:&"
 
_user_specified_name217058:&"
 
_user_specified_name217056:&"
 
_user_specified_name217051:&"
 
_user_specified_name217049:&"
 
_user_specified_name217047:&"
 
_user_specified_name217045:&"
 
_user_specified_name217043:&"
 
_user_specified_name217041:&
"
 
_user_specified_name217039:&	"
 
_user_specified_name217037:&"
 
_user_specified_name216996:&"
 
_user_specified_name216994:&"
 
_user_specified_name216991:&"
 
_user_specified_name216989:&"
 
_user_specified_name216980:&"
 
_user_specified_name216978:&"
 
_user_specified_name216975:&"
 
_user_specified_name216973:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_216827

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_217358
input_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	� 
	unknown_7:�@
	unknown_8:@ 
	unknown_9:�@

unknown_10:@!

unknown_11:�@

unknown_12:@!

unknown_13:@�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_216729p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name217354:&"
 
_user_specified_name217352:&"
 
_user_specified_name217350:&"
 
_user_specified_name217348:&"
 
_user_specified_name217346:&"
 
_user_specified_name217344:&"
 
_user_specified_name217342:&"
 
_user_specified_name217340:&"
 
_user_specified_name217338:&"
 
_user_specified_name217336:&"
 
_user_specified_name217334:&"
 
_user_specified_name217332:&
"
 
_user_specified_name217330:&	"
 
_user_specified_name217328:&"
 
_user_specified_name217326:&"
 
_user_specified_name217324:&"
 
_user_specified_name217322:&"
 
_user_specified_name217320:&"
 
_user_specified_name217318:&"
 
_user_specified_name217316:&"
 
_user_specified_name217314:&"
 
_user_specified_name217312:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
6__inference_layer_normalization_1_layer_call_fn_217635

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_216911p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name217631:&"
 
_user_specified_name217629:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_216951

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�O
__inference__traced_save_218153
file_prefixL
8read_disablecopyonread_transformer_denoiser_dense_kernel:
��G
8read_1_disablecopyonread_transformer_denoiser_dense_bias:	�P
<read_2_disablecopyonread_transformer_denoiser_dense_1_kernel:
��I
:read_3_disablecopyonread_transformer_denoiser_dense_1_bias:	�P
<read_4_disablecopyonread_transformer_denoiser_dense_2_kernel:
��I
:read_5_disablecopyonread_transformer_denoiser_dense_2_bias:	�f
Oread_6_disablecopyonread_transformer_denoiser_multi_head_attention_query_kernel:�@_
Mread_7_disablecopyonread_transformer_denoiser_multi_head_attention_query_bias:@d
Mread_8_disablecopyonread_transformer_denoiser_multi_head_attention_key_kernel:�@]
Kread_9_disablecopyonread_transformer_denoiser_multi_head_attention_key_bias:@g
Pread_10_disablecopyonread_transformer_denoiser_multi_head_attention_value_kernel:�@`
Nread_11_disablecopyonread_transformer_denoiser_multi_head_attention_value_bias:@r
[read_12_disablecopyonread_transformer_denoiser_multi_head_attention_attention_output_kernel:@�h
Yread_13_disablecopyonread_transformer_denoiser_multi_head_attention_attention_output_bias:	�Q
=read_14_disablecopyonread_transformer_denoiser_dense_3_kernel:
��J
;read_15_disablecopyonread_transformer_denoiser_dense_3_bias:	�Q
=read_16_disablecopyonread_transformer_denoiser_dense_4_kernel:
��J
;read_17_disablecopyonread_transformer_denoiser_dense_4_bias:	�W
Hread_18_disablecopyonread_transformer_denoiser_layer_normalization_gamma:	�V
Gread_19_disablecopyonread_transformer_denoiser_layer_normalization_beta:	�Y
Jread_20_disablecopyonread_transformer_denoiser_layer_normalization_1_gamma:	�X
Iread_21_disablecopyonread_transformer_denoiser_layer_normalization_1_beta:	�-
#read_22_disablecopyonread_iteration:	 1
'read_23_disablecopyonread_learning_rate: V
Bread_24_disablecopyonread_adam_m_transformer_denoiser_dense_kernel:
��V
Bread_25_disablecopyonread_adam_v_transformer_denoiser_dense_kernel:
��O
@read_26_disablecopyonread_adam_m_transformer_denoiser_dense_bias:	�O
@read_27_disablecopyonread_adam_v_transformer_denoiser_dense_bias:	�X
Dread_28_disablecopyonread_adam_m_transformer_denoiser_dense_1_kernel:
��X
Dread_29_disablecopyonread_adam_v_transformer_denoiser_dense_1_kernel:
��Q
Bread_30_disablecopyonread_adam_m_transformer_denoiser_dense_1_bias:	�Q
Bread_31_disablecopyonread_adam_v_transformer_denoiser_dense_1_bias:	�X
Dread_32_disablecopyonread_adam_m_transformer_denoiser_dense_2_kernel:
��X
Dread_33_disablecopyonread_adam_v_transformer_denoiser_dense_2_kernel:
��Q
Bread_34_disablecopyonread_adam_m_transformer_denoiser_dense_2_bias:	�Q
Bread_35_disablecopyonread_adam_v_transformer_denoiser_dense_2_bias:	�n
Wread_36_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_query_kernel:�@n
Wread_37_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_query_kernel:�@g
Uread_38_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_query_bias:@g
Uread_39_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_query_bias:@l
Uread_40_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_key_kernel:�@l
Uread_41_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_key_kernel:�@e
Sread_42_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_key_bias:@e
Sread_43_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_key_bias:@n
Wread_44_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_value_kernel:�@n
Wread_45_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_value_kernel:�@g
Uread_46_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_value_bias:@g
Uread_47_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_value_bias:@y
bread_48_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_attention_output_kernel:@�y
bread_49_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_attention_output_kernel:@�o
`read_50_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_attention_output_bias:	�o
`read_51_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_attention_output_bias:	�X
Dread_52_disablecopyonread_adam_m_transformer_denoiser_dense_3_kernel:
��X
Dread_53_disablecopyonread_adam_v_transformer_denoiser_dense_3_kernel:
��Q
Bread_54_disablecopyonread_adam_m_transformer_denoiser_dense_3_bias:	�Q
Bread_55_disablecopyonread_adam_v_transformer_denoiser_dense_3_bias:	�X
Dread_56_disablecopyonread_adam_m_transformer_denoiser_dense_4_kernel:
��X
Dread_57_disablecopyonread_adam_v_transformer_denoiser_dense_4_kernel:
��Q
Bread_58_disablecopyonread_adam_m_transformer_denoiser_dense_4_bias:	�Q
Bread_59_disablecopyonread_adam_v_transformer_denoiser_dense_4_bias:	�^
Oread_60_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_gamma:	�^
Oread_61_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_gamma:	�]
Nread_62_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_beta:	�]
Nread_63_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_beta:	�`
Qread_64_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_1_gamma:	�`
Qread_65_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_1_gamma:	�_
Pread_66_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_1_beta:	�_
Pread_67_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_1_beta:	�)
read_68_disablecopyonread_total: )
read_69_disablecopyonread_count: 
savev2_const
identity_141��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead8read_disablecopyonread_transformer_denoiser_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp8read_disablecopyonread_transformer_denoiser_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_1/DisableCopyOnReadDisableCopyOnRead8read_1_disablecopyonread_transformer_denoiser_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp8read_1_disablecopyonread_transformer_denoiser_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_2/DisableCopyOnReadDisableCopyOnRead<read_2_disablecopyonread_transformer_denoiser_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp<read_2_disablecopyonread_transformer_denoiser_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_3/DisableCopyOnReadDisableCopyOnRead:read_3_disablecopyonread_transformer_denoiser_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp:read_3_disablecopyonread_transformer_denoiser_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_4/DisableCopyOnReadDisableCopyOnRead<read_4_disablecopyonread_transformer_denoiser_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp<read_4_disablecopyonread_transformer_denoiser_dense_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_5/DisableCopyOnReadDisableCopyOnRead:read_5_disablecopyonread_transformer_denoiser_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp:read_5_disablecopyonread_transformer_denoiser_dense_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnReadOread_6_disablecopyonread_transformer_denoiser_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpOread_6_disablecopyonread_transformer_denoiser_multi_head_attention_query_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�@*
dtype0s
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�@j
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_7/DisableCopyOnReadDisableCopyOnReadMread_7_disablecopyonread_transformer_denoiser_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpMread_7_disablecopyonread_transformer_denoiser_multi_head_attention_query_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_8/DisableCopyOnReadDisableCopyOnReadMread_8_disablecopyonread_transformer_denoiser_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpMread_8_disablecopyonread_transformer_denoiser_multi_head_attention_key_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�@*
dtype0s
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�@j
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_9/DisableCopyOnReadDisableCopyOnReadKread_9_disablecopyonread_transformer_denoiser_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpKread_9_disablecopyonread_transformer_denoiser_multi_head_attention_key_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_10/DisableCopyOnReadDisableCopyOnReadPread_10_disablecopyonread_transformer_denoiser_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpPread_10_disablecopyonread_transformer_denoiser_multi_head_attention_value_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�@*
dtype0t
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�@j
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_11/DisableCopyOnReadDisableCopyOnReadNread_11_disablecopyonread_transformer_denoiser_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpNread_11_disablecopyonread_transformer_denoiser_multi_head_attention_value_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_12/DisableCopyOnReadDisableCopyOnRead[read_12_disablecopyonread_transformer_denoiser_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp[read_12_disablecopyonread_transformer_denoiser_multi_head_attention_attention_output_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:@�*
dtype0t
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:@�j
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*#
_output_shapes
:@��
Read_13/DisableCopyOnReadDisableCopyOnReadYread_13_disablecopyonread_transformer_denoiser_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpYread_13_disablecopyonread_transformer_denoiser_multi_head_attention_attention_output_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_14/DisableCopyOnReadDisableCopyOnRead=read_14_disablecopyonread_transformer_denoiser_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp=read_14_disablecopyonread_transformer_denoiser_dense_3_kernel^Read_14/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_15/DisableCopyOnReadDisableCopyOnRead;read_15_disablecopyonread_transformer_denoiser_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp;read_15_disablecopyonread_transformer_denoiser_dense_3_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnRead=read_16_disablecopyonread_transformer_denoiser_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp=read_16_disablecopyonread_transformer_denoiser_dense_4_kernel^Read_16/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_17/DisableCopyOnReadDisableCopyOnRead;read_17_disablecopyonread_transformer_denoiser_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp;read_17_disablecopyonread_transformer_denoiser_dense_4_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnReadHread_18_disablecopyonread_transformer_denoiser_layer_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpHread_18_disablecopyonread_transformer_denoiser_layer_normalization_gamma^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_19/DisableCopyOnReadDisableCopyOnReadGread_19_disablecopyonread_transformer_denoiser_layer_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpGread_19_disablecopyonread_transformer_denoiser_layer_normalization_beta^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnReadJread_20_disablecopyonread_transformer_denoiser_layer_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOpJread_20_disablecopyonread_transformer_denoiser_layer_normalization_1_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_21/DisableCopyOnReadDisableCopyOnReadIread_21_disablecopyonread_transformer_denoiser_layer_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOpIread_21_disablecopyonread_transformer_denoiser_layer_normalization_1_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:�x
Read_22/DisableCopyOnReadDisableCopyOnRead#read_22_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp#read_22_disablecopyonread_iteration^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_learning_rate^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_24/DisableCopyOnReadDisableCopyOnReadBread_24_disablecopyonread_adam_m_transformer_denoiser_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpBread_24_disablecopyonread_adam_m_transformer_denoiser_dense_kernel^Read_24/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_25/DisableCopyOnReadDisableCopyOnReadBread_25_disablecopyonread_adam_v_transformer_denoiser_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOpBread_25_disablecopyonread_adam_v_transformer_denoiser_dense_kernel^Read_25/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_26/DisableCopyOnReadDisableCopyOnRead@read_26_disablecopyonread_adam_m_transformer_denoiser_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp@read_26_disablecopyonread_adam_m_transformer_denoiser_dense_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_27/DisableCopyOnReadDisableCopyOnRead@read_27_disablecopyonread_adam_v_transformer_denoiser_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp@read_27_disablecopyonread_adam_v_transformer_denoiser_dense_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_28/DisableCopyOnReadDisableCopyOnReadDread_28_disablecopyonread_adam_m_transformer_denoiser_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOpDread_28_disablecopyonread_adam_m_transformer_denoiser_dense_1_kernel^Read_28/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_29/DisableCopyOnReadDisableCopyOnReadDread_29_disablecopyonread_adam_v_transformer_denoiser_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpDread_29_disablecopyonread_adam_v_transformer_denoiser_dense_1_kernel^Read_29/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_30/DisableCopyOnReadDisableCopyOnReadBread_30_disablecopyonread_adam_m_transformer_denoiser_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpBread_30_disablecopyonread_adam_m_transformer_denoiser_dense_1_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_31/DisableCopyOnReadDisableCopyOnReadBread_31_disablecopyonread_adam_v_transformer_denoiser_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpBread_31_disablecopyonread_adam_v_transformer_denoiser_dense_1_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_32/DisableCopyOnReadDisableCopyOnReadDread_32_disablecopyonread_adam_m_transformer_denoiser_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOpDread_32_disablecopyonread_adam_m_transformer_denoiser_dense_2_kernel^Read_32/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_33/DisableCopyOnReadDisableCopyOnReadDread_33_disablecopyonread_adam_v_transformer_denoiser_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpDread_33_disablecopyonread_adam_v_transformer_denoiser_dense_2_kernel^Read_33/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_34/DisableCopyOnReadDisableCopyOnReadBread_34_disablecopyonread_adam_m_transformer_denoiser_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOpBread_34_disablecopyonread_adam_m_transformer_denoiser_dense_2_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_35/DisableCopyOnReadDisableCopyOnReadBread_35_disablecopyonread_adam_v_transformer_denoiser_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOpBread_35_disablecopyonread_adam_v_transformer_denoiser_dense_2_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_36/DisableCopyOnReadDisableCopyOnReadWread_36_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOpWread_36_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_query_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�@*
dtype0t
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�@j
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_37/DisableCopyOnReadDisableCopyOnReadWread_37_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOpWread_37_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_query_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�@*
dtype0t
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�@j
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_38/DisableCopyOnReadDisableCopyOnReadUread_38_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOpUread_38_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_query_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_39/DisableCopyOnReadDisableCopyOnReadUread_39_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpUread_39_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_query_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_40/DisableCopyOnReadDisableCopyOnReadUread_40_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpUread_40_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_key_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�@*
dtype0t
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�@j
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_41/DisableCopyOnReadDisableCopyOnReadUread_41_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOpUread_41_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_key_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�@*
dtype0t
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�@j
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_42/DisableCopyOnReadDisableCopyOnReadSread_42_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOpSread_42_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_key_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_43/DisableCopyOnReadDisableCopyOnReadSread_43_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOpSread_43_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_key_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_44/DisableCopyOnReadDisableCopyOnReadWread_44_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpWread_44_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_value_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�@*
dtype0t
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�@j
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_45/DisableCopyOnReadDisableCopyOnReadWread_45_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpWread_45_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_value_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:�@*
dtype0t
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:�@j
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_46/DisableCopyOnReadDisableCopyOnReadUread_46_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOpUread_46_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_value_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_47/DisableCopyOnReadDisableCopyOnReadUread_47_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOpUread_47_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_value_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_48/DisableCopyOnReadDisableCopyOnReadbread_48_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOpbread_48_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_attention_output_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:@�*
dtype0t
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:@�j
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*#
_output_shapes
:@��
Read_49/DisableCopyOnReadDisableCopyOnReadbread_49_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOpbread_49_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_attention_output_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:@�*
dtype0t
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:@�j
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*#
_output_shapes
:@��
Read_50/DisableCopyOnReadDisableCopyOnRead`read_50_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp`read_50_disablecopyonread_adam_m_transformer_denoiser_multi_head_attention_attention_output_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_51/DisableCopyOnReadDisableCopyOnRead`read_51_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp`read_51_disablecopyonread_adam_v_transformer_denoiser_multi_head_attention_attention_output_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_52/DisableCopyOnReadDisableCopyOnReadDread_52_disablecopyonread_adam_m_transformer_denoiser_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpDread_52_disablecopyonread_adam_m_transformer_denoiser_dense_3_kernel^Read_52/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_53/DisableCopyOnReadDisableCopyOnReadDread_53_disablecopyonread_adam_v_transformer_denoiser_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpDread_53_disablecopyonread_adam_v_transformer_denoiser_dense_3_kernel^Read_53/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_54/DisableCopyOnReadDisableCopyOnReadBread_54_disablecopyonread_adam_m_transformer_denoiser_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpBread_54_disablecopyonread_adam_m_transformer_denoiser_dense_3_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_55/DisableCopyOnReadDisableCopyOnReadBread_55_disablecopyonread_adam_v_transformer_denoiser_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOpBread_55_disablecopyonread_adam_v_transformer_denoiser_dense_3_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_56/DisableCopyOnReadDisableCopyOnReadDread_56_disablecopyonread_adam_m_transformer_denoiser_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOpDread_56_disablecopyonread_adam_m_transformer_denoiser_dense_4_kernel^Read_56/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_57/DisableCopyOnReadDisableCopyOnReadDread_57_disablecopyonread_adam_v_transformer_denoiser_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOpDread_57_disablecopyonread_adam_v_transformer_denoiser_dense_4_kernel^Read_57/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_58/DisableCopyOnReadDisableCopyOnReadBread_58_disablecopyonread_adam_m_transformer_denoiser_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOpBread_58_disablecopyonread_adam_m_transformer_denoiser_dense_4_bias^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_59/DisableCopyOnReadDisableCopyOnReadBread_59_disablecopyonread_adam_v_transformer_denoiser_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOpBread_59_disablecopyonread_adam_v_transformer_denoiser_dense_4_bias^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_60/DisableCopyOnReadDisableCopyOnReadOread_60_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOpOread_60_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_gamma^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_61/DisableCopyOnReadDisableCopyOnReadOread_61_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOpOread_61_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_gamma^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_62/DisableCopyOnReadDisableCopyOnReadNread_62_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOpNread_62_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_beta^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_63/DisableCopyOnReadDisableCopyOnReadNread_63_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOpNread_63_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_beta^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_64/DisableCopyOnReadDisableCopyOnReadQread_64_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOpQread_64_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_1_gamma^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_65/DisableCopyOnReadDisableCopyOnReadQread_65_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOpQread_65_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_1_gamma^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_66/DisableCopyOnReadDisableCopyOnReadPread_66_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOpPread_66_disablecopyonread_adam_m_transformer_denoiser_layer_normalization_1_beta^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_67/DisableCopyOnReadDisableCopyOnReadPread_67_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOpPread_67_disablecopyonread_adam_v_transformer_denoiser_layer_normalization_1_beta^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes	
:�t
Read_68/DisableCopyOnReadDisableCopyOnReadread_68_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOpread_68_disablecopyonread_total^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_69/DisableCopyOnReadDisableCopyOnReadread_69_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOpread_69_disablecopyonread_count^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*�
value�B�GB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *U
dtypesK
I2G	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_140Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_141IdentityIdentity_140:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_141Identity_141:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=G9

_output_shapes
: 

_user_specified_nameConst:%F!

_user_specified_namecount:%E!

_user_specified_nametotal:VDR
P
_user_specified_name86Adam/v/transformer_denoiser/layer_normalization_1/beta:VCR
P
_user_specified_name86Adam/m/transformer_denoiser/layer_normalization_1/beta:WBS
Q
_user_specified_name97Adam/v/transformer_denoiser/layer_normalization_1/gamma:WAS
Q
_user_specified_name97Adam/m/transformer_denoiser/layer_normalization_1/gamma:T@P
N
_user_specified_name64Adam/v/transformer_denoiser/layer_normalization/beta:T?P
N
_user_specified_name64Adam/m/transformer_denoiser/layer_normalization/beta:U>Q
O
_user_specified_name75Adam/v/transformer_denoiser/layer_normalization/gamma:U=Q
O
_user_specified_name75Adam/m/transformer_denoiser/layer_normalization/gamma:H<D
B
_user_specified_name*(Adam/v/transformer_denoiser/dense_4/bias:H;D
B
_user_specified_name*(Adam/m/transformer_denoiser/dense_4/bias:J:F
D
_user_specified_name,*Adam/v/transformer_denoiser/dense_4/kernel:J9F
D
_user_specified_name,*Adam/m/transformer_denoiser/dense_4/kernel:H8D
B
_user_specified_name*(Adam/v/transformer_denoiser/dense_3/bias:H7D
B
_user_specified_name*(Adam/m/transformer_denoiser/dense_3/bias:J6F
D
_user_specified_name,*Adam/v/transformer_denoiser/dense_3/kernel:J5F
D
_user_specified_name,*Adam/m/transformer_denoiser/dense_3/kernel:f4b
`
_user_specified_nameHFAdam/v/transformer_denoiser/multi_head_attention/attention_output/bias:f3b
`
_user_specified_nameHFAdam/m/transformer_denoiser/multi_head_attention/attention_output/bias:h2d
b
_user_specified_nameJHAdam/v/transformer_denoiser/multi_head_attention/attention_output/kernel:h1d
b
_user_specified_nameJHAdam/m/transformer_denoiser/multi_head_attention/attention_output/kernel:[0W
U
_user_specified_name=;Adam/v/transformer_denoiser/multi_head_attention/value/bias:[/W
U
_user_specified_name=;Adam/m/transformer_denoiser/multi_head_attention/value/bias:].Y
W
_user_specified_name?=Adam/v/transformer_denoiser/multi_head_attention/value/kernel:]-Y
W
_user_specified_name?=Adam/m/transformer_denoiser/multi_head_attention/value/kernel:Y,U
S
_user_specified_name;9Adam/v/transformer_denoiser/multi_head_attention/key/bias:Y+U
S
_user_specified_name;9Adam/m/transformer_denoiser/multi_head_attention/key/bias:[*W
U
_user_specified_name=;Adam/v/transformer_denoiser/multi_head_attention/key/kernel:[)W
U
_user_specified_name=;Adam/m/transformer_denoiser/multi_head_attention/key/kernel:[(W
U
_user_specified_name=;Adam/v/transformer_denoiser/multi_head_attention/query/bias:['W
U
_user_specified_name=;Adam/m/transformer_denoiser/multi_head_attention/query/bias:]&Y
W
_user_specified_name?=Adam/v/transformer_denoiser/multi_head_attention/query/kernel:]%Y
W
_user_specified_name?=Adam/m/transformer_denoiser/multi_head_attention/query/kernel:H$D
B
_user_specified_name*(Adam/v/transformer_denoiser/dense_2/bias:H#D
B
_user_specified_name*(Adam/m/transformer_denoiser/dense_2/bias:J"F
D
_user_specified_name,*Adam/v/transformer_denoiser/dense_2/kernel:J!F
D
_user_specified_name,*Adam/m/transformer_denoiser/dense_2/kernel:H D
B
_user_specified_name*(Adam/v/transformer_denoiser/dense_1/bias:HD
B
_user_specified_name*(Adam/m/transformer_denoiser/dense_1/bias:JF
D
_user_specified_name,*Adam/v/transformer_denoiser/dense_1/kernel:JF
D
_user_specified_name,*Adam/m/transformer_denoiser/dense_1/kernel:FB
@
_user_specified_name(&Adam/v/transformer_denoiser/dense/bias:FB
@
_user_specified_name(&Adam/m/transformer_denoiser/dense/bias:HD
B
_user_specified_name*(Adam/v/transformer_denoiser/dense/kernel:HD
B
_user_specified_name*(Adam/m/transformer_denoiser/dense/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:OK
I
_user_specified_name1/transformer_denoiser/layer_normalization_1/beta:PL
J
_user_specified_name20transformer_denoiser/layer_normalization_1/gamma:MI
G
_user_specified_name/-transformer_denoiser/layer_normalization/beta:NJ
H
_user_specified_name0.transformer_denoiser/layer_normalization/gamma:A=
;
_user_specified_name#!transformer_denoiser/dense_4/bias:C?
=
_user_specified_name%#transformer_denoiser/dense_4/kernel:A=
;
_user_specified_name#!transformer_denoiser/dense_3/bias:C?
=
_user_specified_name%#transformer_denoiser/dense_3/kernel:_[
Y
_user_specified_nameA?transformer_denoiser/multi_head_attention/attention_output/bias:a]
[
_user_specified_nameCAtransformer_denoiser/multi_head_attention/attention_output/kernel:TP
N
_user_specified_name64transformer_denoiser/multi_head_attention/value/bias:VR
P
_user_specified_name86transformer_denoiser/multi_head_attention/value/kernel:R
N
L
_user_specified_name42transformer_denoiser/multi_head_attention/key/bias:T	P
N
_user_specified_name64transformer_denoiser/multi_head_attention/key/kernel:TP
N
_user_specified_name64transformer_denoiser/multi_head_attention/query/bias:VR
P
_user_specified_name86transformer_denoiser/multi_head_attention/query/kernel:A=
;
_user_specified_name#!transformer_denoiser/dense_2/bias:C?
=
_user_specified_name%#transformer_denoiser/dense_2/kernel:A=
;
_user_specified_name#!transformer_denoiser/dense_1/bias:C?
=
_user_specified_name%#transformer_denoiser/dense_1/kernel:?;
9
_user_specified_name!transformer_denoiser/dense/bias:A=
;
_user_specified_name#!transformer_denoiser/dense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_217626

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5}
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*'
_output_shapes
:���������]
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*'
_output_shapes
:���������
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������s
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*(
_output_shapes
:����������w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0~
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*(
_output_shapes
:����������s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_217576

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:����������T
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:����������O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:����������`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:����������^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�-
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_217036	
query	
value
keyB
+query_einsum_einsum_readvariableop_resource:�@3
!query_add_readvariableop_resource:@@
)key_einsum_einsum_readvariableop_resource:�@1
key_add_readvariableop_resource:@B
+value_einsum_einsum_readvariableop_resource:�@3
!value_add_readvariableop_resource:@M
6attention_output_einsum_einsum_readvariableop_resource:@�;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:QM
,
_output_shapes
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
<
input_11
serving_default_input_1:0����������=
output_11
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2
	
proj1
	attention

dense3

dense4
	norm1
	norm2
dropout1
dropout2
	optimizer

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
/trace_0
0trace_12�
5__inference_transformer_denoiser_layer_call_fn_217128
5__inference_transformer_denoiser_layer_call_fn_217177�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z/trace_0z0trace_1
�
1trace_0
2trace_12�
P__inference_transformer_denoiser_layer_call_and_return_conditional_losses_216970
P__inference_transformer_denoiser_layer_call_and_return_conditional_losses_217079�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z1trace_0z2trace_1
�B�
!__inference__wrapped_model_216729input_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
K_query_dense
L
_key_dense
M_value_dense
N_softmax
O_dropout_layer
P_output_dense"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses
caxis
	&gamma
'beta"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
jaxis
	(gamma
)beta"
_tf_keras_layer
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
q_random_generator"
_tf_keras_layer
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
x_random_generator"
_tf_keras_layer
�
y
_variables
z_iterations
{_learning_rate
|_index_dict
}
_momentums
~_velocities
_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
5:3
��2!transformer_denoiser/dense/kernel
.:,�2transformer_denoiser/dense/bias
7:5
��2#transformer_denoiser/dense_1/kernel
0:.�2!transformer_denoiser/dense_1/bias
7:5
��2#transformer_denoiser/dense_2/kernel
0:.�2!transformer_denoiser/dense_2/bias
M:K�@26transformer_denoiser/multi_head_attention/query/kernel
F:D@24transformer_denoiser/multi_head_attention/query/bias
K:I�@24transformer_denoiser/multi_head_attention/key/kernel
D:B@22transformer_denoiser/multi_head_attention/key/bias
M:K�@26transformer_denoiser/multi_head_attention/value/kernel
F:D@24transformer_denoiser/multi_head_attention/value/bias
X:V@�2Atransformer_denoiser/multi_head_attention/attention_output/kernel
N:L�2?transformer_denoiser/multi_head_attention/attention_output/bias
7:5
��2#transformer_denoiser/dense_3/kernel
0:.�2!transformer_denoiser/dense_3/bias
7:5
��2#transformer_denoiser/dense_4/kernel
0:.�2!transformer_denoiser/dense_4/bias
=:;�2.transformer_denoiser/layer_normalization/gamma
<::�2-transformer_denoiser/layer_normalization/beta
?:=�20transformer_denoiser/layer_normalization_1/gamma
>:<�2/transformer_denoiser/layer_normalization_1/beta
 "
trackable_list_wrapper
f
0
	1

2
3
4
5
6
7
8
9"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_transformer_denoiser_layer_call_fn_217128input_1"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_transformer_denoiser_layer_call_fn_217177input_1"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_transformer_denoiser_layer_call_and_return_conditional_losses_216970input_1"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_transformer_denoiser_layer_call_and_return_conditional_losses_217079input_1"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_layer_call_fn_217367�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_217385�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_1_layer_call_fn_217394�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_217412�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_2_layer_call_fn_217421�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_217431�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
X
0
1
2
3
4
5
 6
!7"
trackable_list_wrapper
X
0
1
2
3
4
5
 6
!7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_multi_head_attention_layer_call_fn_217454
5__inference_multi_head_attention_layer_call_fn_217477�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_217513
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_217549�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

 kernel
!bias"
_tf_keras_layer
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_3_layer_call_fn_217558�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_3_layer_call_and_return_conditional_losses_217576�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_4_layer_call_fn_217585�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_4_layer_call_and_return_conditional_losses_217595�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_layer_normalization_layer_call_fn_217604�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_217626�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_1_layer_call_fn_217635�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_217657�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_layer_call_fn_217662
(__inference_dropout_layer_call_fn_217667�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_layer_call_and_return_conditional_losses_217679
C__inference_dropout_layer_call_and_return_conditional_losses_217684�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_1_layer_call_fn_217689
*__inference_dropout_1_layer_call_fn_217694�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_1_layer_call_and_return_conditional_losses_217706
E__inference_dropout_1_layer_call_and_return_conditional_losses_217711�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
�
z0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_217358input_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_1
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_layer_call_fn_217367inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_layer_call_and_return_conditional_losses_217385inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_1_layer_call_fn_217394inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_217412inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_2_layer_call_fn_217421inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_217431inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
J
K0
L1
M2
N3
O4
P5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_multi_head_attention_layer_call_fn_217454queryvaluekey"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_multi_head_attention_layer_call_fn_217477queryvaluekey"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_217513queryvaluekey"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_217549queryvaluekey"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_3_layer_call_fn_217558inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_3_layer_call_and_return_conditional_losses_217576inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_4_layer_call_fn_217585inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_4_layer_call_and_return_conditional_losses_217595inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_layer_normalization_layer_call_fn_217604inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_217626inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_layer_normalization_1_layer_call_fn_217635inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_217657inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dropout_layer_call_fn_217662inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_layer_call_fn_217667inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_217679inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_217684inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_1_layer_call_fn_217689inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_1_layer_call_fn_217694inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_1_layer_call_and_return_conditional_losses_217706inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_1_layer_call_and_return_conditional_losses_217711inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
::8
��2(Adam/m/transformer_denoiser/dense/kernel
::8
��2(Adam/v/transformer_denoiser/dense/kernel
3:1�2&Adam/m/transformer_denoiser/dense/bias
3:1�2&Adam/v/transformer_denoiser/dense/bias
<::
��2*Adam/m/transformer_denoiser/dense_1/kernel
<::
��2*Adam/v/transformer_denoiser/dense_1/kernel
5:3�2(Adam/m/transformer_denoiser/dense_1/bias
5:3�2(Adam/v/transformer_denoiser/dense_1/bias
<::
��2*Adam/m/transformer_denoiser/dense_2/kernel
<::
��2*Adam/v/transformer_denoiser/dense_2/kernel
5:3�2(Adam/m/transformer_denoiser/dense_2/bias
5:3�2(Adam/v/transformer_denoiser/dense_2/bias
R:P�@2=Adam/m/transformer_denoiser/multi_head_attention/query/kernel
R:P�@2=Adam/v/transformer_denoiser/multi_head_attention/query/kernel
K:I@2;Adam/m/transformer_denoiser/multi_head_attention/query/bias
K:I@2;Adam/v/transformer_denoiser/multi_head_attention/query/bias
P:N�@2;Adam/m/transformer_denoiser/multi_head_attention/key/kernel
P:N�@2;Adam/v/transformer_denoiser/multi_head_attention/key/kernel
I:G@29Adam/m/transformer_denoiser/multi_head_attention/key/bias
I:G@29Adam/v/transformer_denoiser/multi_head_attention/key/bias
R:P�@2=Adam/m/transformer_denoiser/multi_head_attention/value/kernel
R:P�@2=Adam/v/transformer_denoiser/multi_head_attention/value/kernel
K:I@2;Adam/m/transformer_denoiser/multi_head_attention/value/bias
K:I@2;Adam/v/transformer_denoiser/multi_head_attention/value/bias
]:[@�2HAdam/m/transformer_denoiser/multi_head_attention/attention_output/kernel
]:[@�2HAdam/v/transformer_denoiser/multi_head_attention/attention_output/kernel
S:Q�2FAdam/m/transformer_denoiser/multi_head_attention/attention_output/bias
S:Q�2FAdam/v/transformer_denoiser/multi_head_attention/attention_output/bias
<::
��2*Adam/m/transformer_denoiser/dense_3/kernel
<::
��2*Adam/v/transformer_denoiser/dense_3/kernel
5:3�2(Adam/m/transformer_denoiser/dense_3/bias
5:3�2(Adam/v/transformer_denoiser/dense_3/bias
<::
��2*Adam/m/transformer_denoiser/dense_4/kernel
<::
��2*Adam/v/transformer_denoiser/dense_4/kernel
5:3�2(Adam/m/transformer_denoiser/dense_4/bias
5:3�2(Adam/v/transformer_denoiser/dense_4/bias
B:@�25Adam/m/transformer_denoiser/layer_normalization/gamma
B:@�25Adam/v/transformer_denoiser/layer_normalization/gamma
A:?�24Adam/m/transformer_denoiser/layer_normalization/beta
A:?�24Adam/v/transformer_denoiser/layer_normalization/beta
D:B�27Adam/m/transformer_denoiser/layer_normalization_1/gamma
D:B�27Adam/v/transformer_denoiser/layer_normalization_1/gamma
C:A�26Adam/m/transformer_denoiser/layer_normalization_1/beta
C:A�26Adam/v/transformer_denoiser/layer_normalization_1/beta
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
!__inference__wrapped_model_216729�&' !()"#$%1�.
'�$
"�
input_1����������
� "4�1
/
output_1#� 
output_1�����������
C__inference_dense_1_layer_call_and_return_conditional_losses_217412e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_1_layer_call_fn_217394Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_dense_2_layer_call_and_return_conditional_losses_217431e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_2_layer_call_fn_217421Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_dense_3_layer_call_and_return_conditional_losses_217576e"#0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_3_layer_call_fn_217558Z"#0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_dense_4_layer_call_and_return_conditional_losses_217595e$%0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_4_layer_call_fn_217585Z$%0�-
&�#
!�
inputs����������
� ""�
unknown�����������
A__inference_dense_layer_call_and_return_conditional_losses_217385e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_layer_call_fn_217367Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
E__inference_dropout_1_layer_call_and_return_conditional_losses_217706e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
E__inference_dropout_1_layer_call_and_return_conditional_losses_217711e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
*__inference_dropout_1_layer_call_fn_217689Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
*__inference_dropout_1_layer_call_fn_217694Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
C__inference_dropout_layer_call_and_return_conditional_losses_217679e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
C__inference_dropout_layer_call_and_return_conditional_losses_217684e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
(__inference_dropout_layer_call_fn_217662Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
(__inference_dropout_layer_call_fn_217667Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_217657e()0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
6__inference_layer_normalization_1_layer_call_fn_217635Z()0�-
&�#
!�
inputs����������
� ""�
unknown�����������
O__inference_layer_normalization_layer_call_and_return_conditional_losses_217626e&'0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
4__inference_layer_normalization_layer_call_fn_217604Z&'0�-
&�#
!�
inputs����������
� ""�
unknown�����������
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_217513� !���
���
$�!
query����������
$�!
value����������
"�
key����������

 
p 
p
p 
� "1�.
'�$
tensor_0����������
� �
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_217549� !���
���
$�!
query����������
$�!
value����������
"�
key����������

 
p 
p 
p 
� "1�.
'�$
tensor_0����������
� �
5__inference_multi_head_attention_layer_call_fn_217454� !���
���
$�!
query����������
$�!
value����������
"�
key����������

 
p 
p
p 
� "&�#
unknown�����������
5__inference_multi_head_attention_layer_call_fn_217477� !���
���
$�!
query����������
$�!
value����������
"�
key����������

 
p 
p 
p 
� "&�#
unknown�����������
$__inference_signature_wrapper_217358�&' !()"#$%<�9
� 
2�/
-
input_1"�
input_1����������"4�1
/
output_1#� 
output_1�����������
P__inference_transformer_denoiser_layer_call_and_return_conditional_losses_216970~&' !()"#$%5�2
+�(
"�
input_1����������
p
� "-�*
#� 
tensor_0����������
� �
P__inference_transformer_denoiser_layer_call_and_return_conditional_losses_217079~&' !()"#$%5�2
+�(
"�
input_1����������
p 
� "-�*
#� 
tensor_0����������
� �
5__inference_transformer_denoiser_layer_call_fn_217128s&' !()"#$%5�2
+�(
"�
input_1����������
p
� ""�
unknown�����������
5__inference_transformer_denoiser_layer_call_fn_217177s&' !()"#$%5�2
+�(
"�
input_1����������
p 
� ""�
unknown����������