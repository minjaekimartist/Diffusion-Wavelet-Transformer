��
��
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8��
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
=Adam/v/audio_diffusion_conditioner/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*N
shared_name?=Adam/v/audio_diffusion_conditioner/layer_normalization_2/beta
�
QAdam/v/audio_diffusion_conditioner/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp=Adam/v/audio_diffusion_conditioner/layer_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
=Adam/m/audio_diffusion_conditioner/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*N
shared_name?=Adam/m/audio_diffusion_conditioner/layer_normalization_2/beta
�
QAdam/m/audio_diffusion_conditioner/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp=Adam/m/audio_diffusion_conditioner/layer_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
>Adam/v/audio_diffusion_conditioner/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*O
shared_name@>Adam/v/audio_diffusion_conditioner/layer_normalization_2/gamma
�
RAdam/v/audio_diffusion_conditioner/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp>Adam/v/audio_diffusion_conditioner/layer_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
>Adam/m/audio_diffusion_conditioner/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*O
shared_name@>Adam/m/audio_diffusion_conditioner/layer_normalization_2/gamma
�
RAdam/m/audio_diffusion_conditioner/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp>Adam/m/audio_diffusion_conditioner/layer_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
/Adam/v/audio_diffusion_conditioner/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*@
shared_name1/Adam/v/audio_diffusion_conditioner/dense_9/bias
�
CAdam/v/audio_diffusion_conditioner/dense_9/bias/Read/ReadVariableOpReadVariableOp/Adam/v/audio_diffusion_conditioner/dense_9/bias*
_output_shapes	
:�*
dtype0
�
/Adam/m/audio_diffusion_conditioner/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*@
shared_name1/Adam/m/audio_diffusion_conditioner/dense_9/bias
�
CAdam/m/audio_diffusion_conditioner/dense_9/bias/Read/ReadVariableOpReadVariableOp/Adam/m/audio_diffusion_conditioner/dense_9/bias*
_output_shapes	
:�*
dtype0
�
1Adam/v/audio_diffusion_conditioner/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*B
shared_name31Adam/v/audio_diffusion_conditioner/dense_9/kernel
�
EAdam/v/audio_diffusion_conditioner/dense_9/kernel/Read/ReadVariableOpReadVariableOp1Adam/v/audio_diffusion_conditioner/dense_9/kernel* 
_output_shapes
:
��*
dtype0
�
1Adam/m/audio_diffusion_conditioner/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*B
shared_name31Adam/m/audio_diffusion_conditioner/dense_9/kernel
�
EAdam/m/audio_diffusion_conditioner/dense_9/kernel/Read/ReadVariableOpReadVariableOp1Adam/m/audio_diffusion_conditioner/dense_9/kernel* 
_output_shapes
:
��*
dtype0
�
/Adam/v/audio_diffusion_conditioner/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*@
shared_name1/Adam/v/audio_diffusion_conditioner/dense_8/bias
�
CAdam/v/audio_diffusion_conditioner/dense_8/bias/Read/ReadVariableOpReadVariableOp/Adam/v/audio_diffusion_conditioner/dense_8/bias*
_output_shapes	
:�*
dtype0
�
/Adam/m/audio_diffusion_conditioner/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*@
shared_name1/Adam/m/audio_diffusion_conditioner/dense_8/bias
�
CAdam/m/audio_diffusion_conditioner/dense_8/bias/Read/ReadVariableOpReadVariableOp/Adam/m/audio_diffusion_conditioner/dense_8/bias*
_output_shapes	
:�*
dtype0
�
1Adam/v/audio_diffusion_conditioner/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*B
shared_name31Adam/v/audio_diffusion_conditioner/dense_8/kernel
�
EAdam/v/audio_diffusion_conditioner/dense_8/kernel/Read/ReadVariableOpReadVariableOp1Adam/v/audio_diffusion_conditioner/dense_8/kernel* 
_output_shapes
:
��*
dtype0
�
1Adam/m/audio_diffusion_conditioner/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*B
shared_name31Adam/m/audio_diffusion_conditioner/dense_8/kernel
�
EAdam/m/audio_diffusion_conditioner/dense_8/kernel/Read/ReadVariableOpReadVariableOp1Adam/m/audio_diffusion_conditioner/dense_8/kernel* 
_output_shapes
:
��*
dtype0
�
OAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*`
shared_nameQOAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias
�
cAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias/Read/ReadVariableOpReadVariableOpOAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias*
_output_shapes	
:�*
dtype0
�
OAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*`
shared_nameQOAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias
�
cAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias/Read/ReadVariableOpReadVariableOpOAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias*
_output_shapes	
:�*
dtype0
�
QAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*b
shared_nameSQAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel
�
eAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOpQAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel*#
_output_shapes
: �*
dtype0
�
QAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*b
shared_nameSQAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel
�
eAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOpQAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel*#
_output_shapes
: �*
dtype0
�
DAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *U
shared_nameFDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/bias
�
XAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOpDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/bias*
_output_shapes

: *
dtype0
�
DAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *U
shared_nameFDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/bias
�
XAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOpDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/bias*
_output_shapes

: *
dtype0
�
FAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *W
shared_nameHFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/kernel
�
ZAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOpFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/kernel*#
_output_shapes
:� *
dtype0
�
FAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *W
shared_nameHFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/kernel
�
ZAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOpFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/kernel*#
_output_shapes
:� *
dtype0
�
BAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/bias
�
VAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOpBAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/bias*
_output_shapes

: *
dtype0
�
BAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/bias
�
VAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOpBAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/bias*
_output_shapes

: *
dtype0
�
DAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *U
shared_nameFDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/kernel
�
XAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOpDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/kernel*#
_output_shapes
:� *
dtype0
�
DAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *U
shared_nameFDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/kernel
�
XAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOpDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/kernel*#
_output_shapes
:� *
dtype0
�
DAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *U
shared_nameFDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/bias
�
XAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOpDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/bias*
_output_shapes

: *
dtype0
�
DAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *U
shared_nameFDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/bias
�
XAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOpDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/bias*
_output_shapes

: *
dtype0
�
FAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *W
shared_nameHFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/kernel
�
ZAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOpFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/kernel*#
_output_shapes
:� *
dtype0
�
FAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *W
shared_nameHFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/kernel
�
ZAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOpFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/kernel*#
_output_shapes
:� *
dtype0

Adam/v/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/dense_7/bias
x
'Adam/v/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/bias*
_output_shapes	
:�*
dtype0

Adam/m/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/dense_7/bias
x
'Adam/m/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/dense_7/kernel
�
)Adam/v/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/dense_7/kernel
�
)Adam/m/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/kernel* 
_output_shapes
:
��*
dtype0

Adam/v/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/dense_6/bias
x
'Adam/v/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/bias*
_output_shapes	
:�*
dtype0

Adam/m/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/dense_6/bias
x
'Adam/m/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/dense_6/kernel
�
)Adam/v/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/dense_6/kernel
�
)Adam/m/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/kernel* 
_output_shapes
:
��*
dtype0

Adam/v/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/dense_5/bias
x
'Adam/v/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/bias*
_output_shapes	
:�*
dtype0

Adam/m/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/dense_5/bias
x
'Adam/m/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/dense_5/kernel
�
)Adam/v/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/dense_5/kernel
�
)Adam/m/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/kernel* 
_output_shapes
:
��*
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
6audio_diffusion_conditioner/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*G
shared_name86audio_diffusion_conditioner/layer_normalization_2/beta
�
Jaudio_diffusion_conditioner/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp6audio_diffusion_conditioner/layer_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
7audio_diffusion_conditioner/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*H
shared_name97audio_diffusion_conditioner/layer_normalization_2/gamma
�
Kaudio_diffusion_conditioner/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp7audio_diffusion_conditioner/layer_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
(audio_diffusion_conditioner/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(audio_diffusion_conditioner/dense_9/bias
�
<audio_diffusion_conditioner/dense_9/bias/Read/ReadVariableOpReadVariableOp(audio_diffusion_conditioner/dense_9/bias*
_output_shapes	
:�*
dtype0
�
*audio_diffusion_conditioner/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*audio_diffusion_conditioner/dense_9/kernel
�
>audio_diffusion_conditioner/dense_9/kernel/Read/ReadVariableOpReadVariableOp*audio_diffusion_conditioner/dense_9/kernel* 
_output_shapes
:
��*
dtype0
�
(audio_diffusion_conditioner/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(audio_diffusion_conditioner/dense_8/bias
�
<audio_diffusion_conditioner/dense_8/bias/Read/ReadVariableOpReadVariableOp(audio_diffusion_conditioner/dense_8/bias*
_output_shapes	
:�*
dtype0
�
*audio_diffusion_conditioner/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*audio_diffusion_conditioner/dense_8/kernel
�
>audio_diffusion_conditioner/dense_8/kernel/Read/ReadVariableOpReadVariableOp*audio_diffusion_conditioner/dense_8/kernel* 
_output_shapes
:
��*
dtype0
�
Haudio_diffusion_conditioner/multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*Y
shared_nameJHaudio_diffusion_conditioner/multi_head_attention_1/attention_output/bias
�
\audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias/Read/ReadVariableOpReadVariableOpHaudio_diffusion_conditioner/multi_head_attention_1/attention_output/bias*
_output_shapes	
:�*
dtype0
�
Jaudio_diffusion_conditioner/multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*[
shared_nameLJaudio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel
�
^audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOpJaudio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel*#
_output_shapes
: �*
dtype0
�
=audio_diffusion_conditioner/multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *N
shared_name?=audio_diffusion_conditioner/multi_head_attention_1/value/bias
�
Qaudio_diffusion_conditioner/multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp=audio_diffusion_conditioner/multi_head_attention_1/value/bias*
_output_shapes

: *
dtype0
�
?audio_diffusion_conditioner/multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *P
shared_nameA?audio_diffusion_conditioner/multi_head_attention_1/value/kernel
�
Saudio_diffusion_conditioner/multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp?audio_diffusion_conditioner/multi_head_attention_1/value/kernel*#
_output_shapes
:� *
dtype0
�
;audio_diffusion_conditioner/multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *L
shared_name=;audio_diffusion_conditioner/multi_head_attention_1/key/bias
�
Oaudio_diffusion_conditioner/multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOp;audio_diffusion_conditioner/multi_head_attention_1/key/bias*
_output_shapes

: *
dtype0
�
=audio_diffusion_conditioner/multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *N
shared_name?=audio_diffusion_conditioner/multi_head_attention_1/key/kernel
�
Qaudio_diffusion_conditioner/multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp=audio_diffusion_conditioner/multi_head_attention_1/key/kernel*#
_output_shapes
:� *
dtype0
�
=audio_diffusion_conditioner/multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *N
shared_name?=audio_diffusion_conditioner/multi_head_attention_1/query/bias
�
Qaudio_diffusion_conditioner/multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp=audio_diffusion_conditioner/multi_head_attention_1/query/bias*
_output_shapes

: *
dtype0
�
?audio_diffusion_conditioner/multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *P
shared_nameA?audio_diffusion_conditioner/multi_head_attention_1/query/kernel
�
Saudio_diffusion_conditioner/multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp?audio_diffusion_conditioner/multi_head_attention_1/query/kernel*#
_output_shapes
:� *
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:�*
dtype0
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
��*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:�*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
��*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:�*
dtype0
z
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
��*
dtype0
|
serving_default_input_1Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias?audio_diffusion_conditioner/multi_head_attention_1/query/kernel=audio_diffusion_conditioner/multi_head_attention_1/query/bias=audio_diffusion_conditioner/multi_head_attention_1/key/kernel;audio_diffusion_conditioner/multi_head_attention_1/key/bias?audio_diffusion_conditioner/multi_head_attention_1/value/kernel=audio_diffusion_conditioner/multi_head_attention_1/value/biasJaudio_diffusion_conditioner/multi_head_attention_1/attention_output/kernelHaudio_diffusion_conditioner/multi_head_attention_1/attention_output/bias7audio_diffusion_conditioner/layer_normalization_2/gamma6audio_diffusion_conditioner/layer_normalization_2/beta*audio_diffusion_conditioner/dense_8/kernel(audio_diffusion_conditioner/dense_8/bias*audio_diffusion_conditioner/dense_9/kernel(audio_diffusion_conditioner/dense_9/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_225750

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ǖ
value��B�� B��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
text_encoder
		attention

frequency_proj
volume_proj
	layernorm
	optimizer

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19*
* 
�
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

(trace_0
)trace_1* 

*trace_0
+trace_1* 
* 
�
,layer_with_weights-0
,layer-0
-layer-1
.layer_with_weights-1
.layer-2
/layer-3
0layer_with_weights-2
0layer-4
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=_query_dense
>
_key_dense
?_value_dense
@_softmax
A_dropout_layer
B_output_dense*
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

kernel
bias*
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

kernel
 bias*
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Uaxis
	!gamma
"beta*
�
V
_variables
W_iterations
X_learning_rate
Y_index_dict
Z
_momentums
[_velocities
\_update_step_xla*

]serving_default* 
NH
VARIABLE_VALUEdense_5/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_5/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_6/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_6/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_7/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_7/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?audio_diffusion_conditioner/multi_head_attention_1/query/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE=audio_diffusion_conditioner/multi_head_attention_1/query/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE=audio_diffusion_conditioner/multi_head_attention_1/key/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE;audio_diffusion_conditioner/multi_head_attention_1/key/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE?audio_diffusion_conditioner/multi_head_attention_1/value/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=audio_diffusion_conditioner/multi_head_attention_1/value/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEJaudio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEHaudio_diffusion_conditioner/multi_head_attention_1/attention_output/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*audio_diffusion_conditioner/dense_8/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(audio_diffusion_conditioner/dense_8/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*audio_diffusion_conditioner/dense_9/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(audio_diffusion_conditioner/dense_9/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7audio_diffusion_conditioner/layer_normalization_2/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6audio_diffusion_conditioner/layer_normalization_2/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
	1

2
3
4*

^0*
* 
* 
* 
* 
* 
* 
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

kernel
bias*
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
k_random_generator* 
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

kernel
bias*
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
x_random_generator* 
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses

kernel
bias*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

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

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape

kernel
bias*
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

kernel
bias*

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
 1*

0
 1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

!0
"1*

!0
"1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
�
W0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19*
* 
* 
<
�	variables
�	keras_api

�total

�count*

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
'
,0
-1
.2
/3
04*
* 
* 
* 
* 
* 
* 
* 
* 
.
=0
>1
?2
@3
A4
B5*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
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
0
1*

0
1*
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
0
1*

0
1*
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
0
1*

0
1*
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
`Z
VARIABLE_VALUEAdam/m/dense_5/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_5/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_5/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_5/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_6/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_6/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_6/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_6/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_7/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_7/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_7/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_7/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEBAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEBAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEQAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEQAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEOAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEOAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE1Adam/m/audio_diffusion_conditioner/dense_8/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE1Adam/v/audio_diffusion_conditioner/dense_8/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE/Adam/m/audio_diffusion_conditioner/dense_8/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE/Adam/v/audio_diffusion_conditioner/dense_8/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE1Adam/m/audio_diffusion_conditioner/dense_9/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE1Adam/v/audio_diffusion_conditioner/dense_9/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE/Adam/m/audio_diffusion_conditioner/dense_9/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE/Adam/v/audio_diffusion_conditioner/dense_9/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE>Adam/m/audio_diffusion_conditioner/layer_normalization_2/gamma2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE>Adam/v/audio_diffusion_conditioner/layer_normalization_2/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE=Adam/m/audio_diffusion_conditioner/layer_normalization_2/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE=Adam/v/audio_diffusion_conditioner/layer_normalization_2/beta2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias?audio_diffusion_conditioner/multi_head_attention_1/query/kernel=audio_diffusion_conditioner/multi_head_attention_1/query/bias=audio_diffusion_conditioner/multi_head_attention_1/key/kernel;audio_diffusion_conditioner/multi_head_attention_1/key/bias?audio_diffusion_conditioner/multi_head_attention_1/value/kernel=audio_diffusion_conditioner/multi_head_attention_1/value/biasJaudio_diffusion_conditioner/multi_head_attention_1/attention_output/kernelHaudio_diffusion_conditioner/multi_head_attention_1/attention_output/bias*audio_diffusion_conditioner/dense_8/kernel(audio_diffusion_conditioner/dense_8/bias*audio_diffusion_conditioner/dense_9/kernel(audio_diffusion_conditioner/dense_9/bias7audio_diffusion_conditioner/layer_normalization_2/gamma6audio_diffusion_conditioner/layer_normalization_2/beta	iterationlearning_rateAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/kernelFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/kernelDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/biasDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/biasDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/kernelDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/kernelBAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/biasBAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/biasFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/kernelFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/kernelDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/biasDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/biasQAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernelQAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernelOAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/biasOAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias1Adam/m/audio_diffusion_conditioner/dense_8/kernel1Adam/v/audio_diffusion_conditioner/dense_8/kernel/Adam/m/audio_diffusion_conditioner/dense_8/bias/Adam/v/audio_diffusion_conditioner/dense_8/bias1Adam/m/audio_diffusion_conditioner/dense_9/kernel1Adam/v/audio_diffusion_conditioner/dense_9/kernel/Adam/m/audio_diffusion_conditioner/dense_9/bias/Adam/v/audio_diffusion_conditioner/dense_9/bias>Adam/m/audio_diffusion_conditioner/layer_normalization_2/gamma>Adam/v/audio_diffusion_conditioner/layer_normalization_2/gamma=Adam/m/audio_diffusion_conditioner/layer_normalization_2/beta=Adam/v/audio_diffusion_conditioner/layer_normalization_2/betatotalcountConst*M
TinF
D2B*
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
__inference__traced_save_226472
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias?audio_diffusion_conditioner/multi_head_attention_1/query/kernel=audio_diffusion_conditioner/multi_head_attention_1/query/bias=audio_diffusion_conditioner/multi_head_attention_1/key/kernel;audio_diffusion_conditioner/multi_head_attention_1/key/bias?audio_diffusion_conditioner/multi_head_attention_1/value/kernel=audio_diffusion_conditioner/multi_head_attention_1/value/biasJaudio_diffusion_conditioner/multi_head_attention_1/attention_output/kernelHaudio_diffusion_conditioner/multi_head_attention_1/attention_output/bias*audio_diffusion_conditioner/dense_8/kernel(audio_diffusion_conditioner/dense_8/bias*audio_diffusion_conditioner/dense_9/kernel(audio_diffusion_conditioner/dense_9/bias7audio_diffusion_conditioner/layer_normalization_2/gamma6audio_diffusion_conditioner/layer_normalization_2/beta	iterationlearning_rateAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/kernelFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/kernelDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/biasDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/biasDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/kernelDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/kernelBAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/biasBAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/biasFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/kernelFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/kernelDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/biasDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/biasQAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernelQAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernelOAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/biasOAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias1Adam/m/audio_diffusion_conditioner/dense_8/kernel1Adam/v/audio_diffusion_conditioner/dense_8/kernel/Adam/m/audio_diffusion_conditioner/dense_8/bias/Adam/v/audio_diffusion_conditioner/dense_8/bias1Adam/m/audio_diffusion_conditioner/dense_9/kernel1Adam/v/audio_diffusion_conditioner/dense_9/kernel/Adam/m/audio_diffusion_conditioner/dense_9/bias/Adam/v/audio_diffusion_conditioner/dense_9/bias>Adam/m/audio_diffusion_conditioner/layer_normalization_2/gamma>Adam/v/audio_diffusion_conditioner/layer_normalization_2/gamma=Adam/m/audio_diffusion_conditioner/layer_normalization_2/beta=Adam/v/audio_diffusion_conditioner/layer_normalization_2/betatotalcount*L
TinE
C2A*
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
"__inference__traced_restore_226673��
�

�
C__inference_dense_7_layer_call_and_return_conditional_losses_225194

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
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
:����������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_225939

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
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
:����������l
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
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������s
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*(
_output_shapes
:����������w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0~
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*(
_output_shapes
:����������s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 24
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
:����������
 
_user_specified_nameinputs
�

�
+__inference_sequential_layer_call_fn_225249
input_1
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_225201p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name225245:&"
 
_user_specified_name225243:&"
 
_user_specified_name225241:&"
 
_user_specified_name225239:&"
 
_user_specified_name225237:&"
 
_user_specified_name225235:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_225407

inputs4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
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
:����������l
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
:�*
dtype0�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������s
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*(
_output_shapes
:����������w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0~
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*(
_output_shapes
:����������s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:����������\
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 24
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
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_3_layer_call_fn_226030

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_225224a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
7__inference_multi_head_attention_1_layer_call_fn_225773	
query	
value
key
unknown:� 
	unknown_0:  
	unknown_1:� 
	unknown_2:  
	unknown_3:� 
	unknown_4:  
	unknown_5: �
	unknown_6:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225366t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&
"
 
_user_specified_name225769:&	"
 
_user_specified_name225767:&"
 
_user_specified_name225765:&"
 
_user_specified_name225763:&"
 
_user_specified_name225761:&"
 
_user_specified_name225759:&"
 
_user_specified_name225757:&"
 
_user_specified_name225755:QM
,
_output_shapes
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery
�
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_226047

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�-
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225366	
query	
value
keyB
+query_einsum_einsum_readvariableop_resource:� 3
!query_add_readvariableop_resource: @
)key_einsum_einsum_readvariableop_resource:� 1
key_add_readvariableop_resource: B
+value_einsum_einsum_readvariableop_resource:� 3
!value_add_readvariableop_resource: M
6attention_output_einsum_einsum_readvariableop_resource: �;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:��������� �
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
:��������� *
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 2J
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
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery
�

�
C__inference_dense_8_layer_call_and_return_conditional_losses_225888

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
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
:����������
 
_user_specified_nameinputs
�0
�	
W__inference_audio_diffusion_conditioner_layer_call_and_return_conditional_losses_225448
input_1%
sequential_225316:
�� 
sequential_225318:	�%
sequential_225320:
�� 
sequential_225322:	�%
sequential_225324:
�� 
sequential_225326:	�4
multi_head_attention_1_225367:� /
multi_head_attention_1_225369: 4
multi_head_attention_1_225371:� /
multi_head_attention_1_225373: 4
multi_head_attention_1_225375:� /
multi_head_attention_1_225377: 4
multi_head_attention_1_225379: �,
multi_head_attention_1_225381:	�+
layer_normalization_2_225408:	�+
layer_normalization_2_225410:	�"
dense_8_225424:
��
dense_8_225426:	�"
dense_9_225440:
��
dense_9_225442:	�
identity��dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�.multi_head_attention_1/StatefulPartitionedCall�"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_225316sequential_225318sequential_225320sequential_225322sequential_225324sequential_225326*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_225201P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�

ExpandDims
ExpandDims+sequential/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:�����������
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0ExpandDims:output:0ExpandDims:output:0multi_head_attention_1_225367multi_head_attention_1_225369multi_head_attention_1_225371multi_head_attention_1_225373multi_head_attention_1_225375multi_head_attention_1_225377multi_head_attention_1_225379multi_head_attention_1_225381*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225366�
SqueezeSqueeze7multi_head_attention_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������*
squeeze_dims
~
addAddV2+sequential/StatefulPartitionedCall:output:0Squeeze:output:0*
T0*(
_output_shapes
:�����������
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd:z:0layer_normalization_2_225408layer_normalization_2_225410*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_225407�
dense_8/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_8_225424dense_8_225426*
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
C__inference_dense_8_layer_call_and_return_conditional_losses_225423�
dense_9/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_9_225440dense_9_225442*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_225439V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
concatConcatV2(dense_8/StatefulPartitionedCall:output:0(dense_9/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:����������	�
NoOpNoOp ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:&"
 
_user_specified_name225442:&"
 
_user_specified_name225440:&"
 
_user_specified_name225426:&"
 
_user_specified_name225424:&"
 
_user_specified_name225410:&"
 
_user_specified_name225408:&"
 
_user_specified_name225381:&"
 
_user_specified_name225379:&"
 
_user_specified_name225377:&"
 
_user_specified_name225375:&
"
 
_user_specified_name225373:&	"
 
_user_specified_name225371:&"
 
_user_specified_name225369:&"
 
_user_specified_name225367:&"
 
_user_specified_name225326:&"
 
_user_specified_name225324:&"
 
_user_specified_name225322:&"
 
_user_specified_name225320:&"
 
_user_specified_name225318:&"
 
_user_specified_name225316:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
��
�
!__inference__wrapped_model_225110
input_1a
Maudio_diffusion_conditioner_sequential_dense_5_matmul_readvariableop_resource:
��]
Naudio_diffusion_conditioner_sequential_dense_5_biasadd_readvariableop_resource:	�a
Maudio_diffusion_conditioner_sequential_dense_6_matmul_readvariableop_resource:
��]
Naudio_diffusion_conditioner_sequential_dense_6_biasadd_readvariableop_resource:	�a
Maudio_diffusion_conditioner_sequential_dense_7_matmul_readvariableop_resource:
��]
Naudio_diffusion_conditioner_sequential_dense_7_biasadd_readvariableop_resource:	�u
^audio_diffusion_conditioner_multi_head_attention_1_query_einsum_einsum_readvariableop_resource:� f
Taudio_diffusion_conditioner_multi_head_attention_1_query_add_readvariableop_resource: s
\audio_diffusion_conditioner_multi_head_attention_1_key_einsum_einsum_readvariableop_resource:� d
Raudio_diffusion_conditioner_multi_head_attention_1_key_add_readvariableop_resource: u
^audio_diffusion_conditioner_multi_head_attention_1_value_einsum_einsum_readvariableop_resource:� f
Taudio_diffusion_conditioner_multi_head_attention_1_value_add_readvariableop_resource: �
iaudio_diffusion_conditioner_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource: �n
_audio_diffusion_conditioner_multi_head_attention_1_attention_output_add_readvariableop_resource:	�f
Waudio_diffusion_conditioner_layer_normalization_2_batchnorm_mul_readvariableop_resource:	�b
Saudio_diffusion_conditioner_layer_normalization_2_batchnorm_readvariableop_resource:	�V
Baudio_diffusion_conditioner_dense_8_matmul_readvariableop_resource:
��R
Caudio_diffusion_conditioner_dense_8_biasadd_readvariableop_resource:	�V
Baudio_diffusion_conditioner_dense_9_matmul_readvariableop_resource:
��R
Caudio_diffusion_conditioner_dense_9_biasadd_readvariableop_resource:	�
identity��:audio_diffusion_conditioner/dense_8/BiasAdd/ReadVariableOp�9audio_diffusion_conditioner/dense_8/MatMul/ReadVariableOp�:audio_diffusion_conditioner/dense_9/BiasAdd/ReadVariableOp�9audio_diffusion_conditioner/dense_9/MatMul/ReadVariableOp�Jaudio_diffusion_conditioner/layer_normalization_2/batchnorm/ReadVariableOp�Naudio_diffusion_conditioner/layer_normalization_2/batchnorm/mul/ReadVariableOp�Vaudio_diffusion_conditioner/multi_head_attention_1/attention_output/add/ReadVariableOp�`audio_diffusion_conditioner/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp�Iaudio_diffusion_conditioner/multi_head_attention_1/key/add/ReadVariableOp�Saudio_diffusion_conditioner/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp�Kaudio_diffusion_conditioner/multi_head_attention_1/query/add/ReadVariableOp�Uaudio_diffusion_conditioner/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp�Kaudio_diffusion_conditioner/multi_head_attention_1/value/add/ReadVariableOp�Uaudio_diffusion_conditioner/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp�Eaudio_diffusion_conditioner/sequential/dense_5/BiasAdd/ReadVariableOp�Daudio_diffusion_conditioner/sequential/dense_5/MatMul/ReadVariableOp�Eaudio_diffusion_conditioner/sequential/dense_6/BiasAdd/ReadVariableOp�Daudio_diffusion_conditioner/sequential/dense_6/MatMul/ReadVariableOp�Eaudio_diffusion_conditioner/sequential/dense_7/BiasAdd/ReadVariableOp�Daudio_diffusion_conditioner/sequential/dense_7/MatMul/ReadVariableOp�
Daudio_diffusion_conditioner/sequential/dense_5/MatMul/ReadVariableOpReadVariableOpMaudio_diffusion_conditioner_sequential_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
5audio_diffusion_conditioner/sequential/dense_5/MatMulMatMulinput_1Laudio_diffusion_conditioner/sequential/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Eaudio_diffusion_conditioner/sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOpNaudio_diffusion_conditioner_sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6audio_diffusion_conditioner/sequential/dense_5/BiasAddBiasAdd?audio_diffusion_conditioner/sequential/dense_5/MatMul:product:0Maudio_diffusion_conditioner/sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������~
9audio_diffusion_conditioner/sequential/dense_5/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
7audio_diffusion_conditioner/sequential/dense_5/Gelu/mulMulBaudio_diffusion_conditioner/sequential/dense_5/Gelu/mul/x:output:0?audio_diffusion_conditioner/sequential/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:����������
:audio_diffusion_conditioner/sequential/dense_5/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
;audio_diffusion_conditioner/sequential/dense_5/Gelu/truedivRealDiv?audio_diffusion_conditioner/sequential/dense_5/BiasAdd:output:0Caudio_diffusion_conditioner/sequential/dense_5/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:�����������
7audio_diffusion_conditioner/sequential/dense_5/Gelu/ErfErf?audio_diffusion_conditioner/sequential/dense_5/Gelu/truediv:z:0*
T0*(
_output_shapes
:����������~
9audio_diffusion_conditioner/sequential/dense_5/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7audio_diffusion_conditioner/sequential/dense_5/Gelu/addAddV2Baudio_diffusion_conditioner/sequential/dense_5/Gelu/add/x:output:0;audio_diffusion_conditioner/sequential/dense_5/Gelu/Erf:y:0*
T0*(
_output_shapes
:�����������
9audio_diffusion_conditioner/sequential/dense_5/Gelu/mul_1Mul;audio_diffusion_conditioner/sequential/dense_5/Gelu/mul:z:0;audio_diffusion_conditioner/sequential/dense_5/Gelu/add:z:0*
T0*(
_output_shapes
:�����������
9audio_diffusion_conditioner/sequential/dropout_2/IdentityIdentity=audio_diffusion_conditioner/sequential/dense_5/Gelu/mul_1:z:0*
T0*(
_output_shapes
:�����������
Daudio_diffusion_conditioner/sequential/dense_6/MatMul/ReadVariableOpReadVariableOpMaudio_diffusion_conditioner_sequential_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
5audio_diffusion_conditioner/sequential/dense_6/MatMulMatMulBaudio_diffusion_conditioner/sequential/dropout_2/Identity:output:0Laudio_diffusion_conditioner/sequential/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Eaudio_diffusion_conditioner/sequential/dense_6/BiasAdd/ReadVariableOpReadVariableOpNaudio_diffusion_conditioner_sequential_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6audio_diffusion_conditioner/sequential/dense_6/BiasAddBiasAdd?audio_diffusion_conditioner/sequential/dense_6/MatMul:product:0Maudio_diffusion_conditioner/sequential/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������~
9audio_diffusion_conditioner/sequential/dense_6/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
7audio_diffusion_conditioner/sequential/dense_6/Gelu/mulMulBaudio_diffusion_conditioner/sequential/dense_6/Gelu/mul/x:output:0?audio_diffusion_conditioner/sequential/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:����������
:audio_diffusion_conditioner/sequential/dense_6/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
;audio_diffusion_conditioner/sequential/dense_6/Gelu/truedivRealDiv?audio_diffusion_conditioner/sequential/dense_6/BiasAdd:output:0Caudio_diffusion_conditioner/sequential/dense_6/Gelu/Cast/x:output:0*
T0*(
_output_shapes
:�����������
7audio_diffusion_conditioner/sequential/dense_6/Gelu/ErfErf?audio_diffusion_conditioner/sequential/dense_6/Gelu/truediv:z:0*
T0*(
_output_shapes
:����������~
9audio_diffusion_conditioner/sequential/dense_6/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7audio_diffusion_conditioner/sequential/dense_6/Gelu/addAddV2Baudio_diffusion_conditioner/sequential/dense_6/Gelu/add/x:output:0;audio_diffusion_conditioner/sequential/dense_6/Gelu/Erf:y:0*
T0*(
_output_shapes
:�����������
9audio_diffusion_conditioner/sequential/dense_6/Gelu/mul_1Mul;audio_diffusion_conditioner/sequential/dense_6/Gelu/mul:z:0;audio_diffusion_conditioner/sequential/dense_6/Gelu/add:z:0*
T0*(
_output_shapes
:�����������
9audio_diffusion_conditioner/sequential/dropout_3/IdentityIdentity=audio_diffusion_conditioner/sequential/dense_6/Gelu/mul_1:z:0*
T0*(
_output_shapes
:�����������
Daudio_diffusion_conditioner/sequential/dense_7/MatMul/ReadVariableOpReadVariableOpMaudio_diffusion_conditioner_sequential_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
5audio_diffusion_conditioner/sequential/dense_7/MatMulMatMulBaudio_diffusion_conditioner/sequential/dropout_3/Identity:output:0Laudio_diffusion_conditioner/sequential/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Eaudio_diffusion_conditioner/sequential/dense_7/BiasAdd/ReadVariableOpReadVariableOpNaudio_diffusion_conditioner_sequential_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
6audio_diffusion_conditioner/sequential/dense_7/BiasAddBiasAdd?audio_diffusion_conditioner/sequential/dense_7/MatMul:product:0Maudio_diffusion_conditioner/sequential/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*audio_diffusion_conditioner/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
&audio_diffusion_conditioner/ExpandDims
ExpandDims?audio_diffusion_conditioner/sequential/dense_7/BiasAdd:output:03audio_diffusion_conditioner/ExpandDims/dim:output:0*
T0*,
_output_shapes
:�����������
Uaudio_diffusion_conditioner/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOp^audio_diffusion_conditioner_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
Faudio_diffusion_conditioner/multi_head_attention_1/query/einsum/EinsumEinsum/audio_diffusion_conditioner/ExpandDims:output:0]audio_diffusion_conditioner/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
Kaudio_diffusion_conditioner/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpTaudio_diffusion_conditioner_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
<audio_diffusion_conditioner/multi_head_attention_1/query/addAddV2Oaudio_diffusion_conditioner/multi_head_attention_1/query/einsum/Einsum:output:0Saudio_diffusion_conditioner/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
Saudio_diffusion_conditioner/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp\audio_diffusion_conditioner_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
Daudio_diffusion_conditioner/multi_head_attention_1/key/einsum/EinsumEinsum/audio_diffusion_conditioner/ExpandDims:output:0[audio_diffusion_conditioner/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
Iaudio_diffusion_conditioner/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpRaudio_diffusion_conditioner_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype0�
:audio_diffusion_conditioner/multi_head_attention_1/key/addAddV2Maudio_diffusion_conditioner/multi_head_attention_1/key/einsum/Einsum:output:0Qaudio_diffusion_conditioner/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
Uaudio_diffusion_conditioner/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOp^audio_diffusion_conditioner_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
Faudio_diffusion_conditioner/multi_head_attention_1/value/einsum/EinsumEinsum/audio_diffusion_conditioner/ExpandDims:output:0]audio_diffusion_conditioner/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abde�
Kaudio_diffusion_conditioner/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpTaudio_diffusion_conditioner_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
<audio_diffusion_conditioner/multi_head_attention_1/value/addAddV2Oaudio_diffusion_conditioner/multi_head_attention_1/value/einsum/Einsum:output:0Saudio_diffusion_conditioner/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� }
8audio_diffusion_conditioner/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>�
6audio_diffusion_conditioner/multi_head_attention_1/MulMul@audio_diffusion_conditioner/multi_head_attention_1/query/add:z:0Aaudio_diffusion_conditioner/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:��������� �
@audio_diffusion_conditioner/multi_head_attention_1/einsum/EinsumEinsum>audio_diffusion_conditioner/multi_head_attention_1/key/add:z:0:audio_diffusion_conditioner/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:���������*
equationaecd,abcd->acbe�
Baudio_diffusion_conditioner/multi_head_attention_1/softmax/SoftmaxSoftmaxIaudio_diffusion_conditioner/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:����������
Caudio_diffusion_conditioner/multi_head_attention_1/dropout/IdentityIdentityLaudio_diffusion_conditioner/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:����������
Baudio_diffusion_conditioner/multi_head_attention_1/einsum_1/EinsumEinsumLaudio_diffusion_conditioner/multi_head_attention_1/dropout/Identity:output:0@audio_diffusion_conditioner/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:��������� *
equationacbe,aecd->abcd�
`audio_diffusion_conditioner/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpiaudio_diffusion_conditioner_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
Qaudio_diffusion_conditioner/multi_head_attention_1/attention_output/einsum/EinsumEinsumKaudio_diffusion_conditioner/multi_head_attention_1/einsum_1/Einsum:output:0haudio_diffusion_conditioner/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
Vaudio_diffusion_conditioner/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOp_audio_diffusion_conditioner_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gaudio_diffusion_conditioner/multi_head_attention_1/attention_output/addAddV2Zaudio_diffusion_conditioner/multi_head_attention_1/attention_output/einsum/Einsum:output:0^audio_diffusion_conditioner/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
#audio_diffusion_conditioner/SqueezeSqueezeKaudio_diffusion_conditioner/multi_head_attention_1/attention_output/add:z:0*
T0*(
_output_shapes
:����������*
squeeze_dims
�
audio_diffusion_conditioner/addAddV2?audio_diffusion_conditioner/sequential/dense_7/BiasAdd:output:0,audio_diffusion_conditioner/Squeeze:output:0*
T0*(
_output_shapes
:�����������
Paudio_diffusion_conditioner/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
>audio_diffusion_conditioner/layer_normalization_2/moments/meanMean#audio_diffusion_conditioner/add:z:0Yaudio_diffusion_conditioner/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(�
Faudio_diffusion_conditioner/layer_normalization_2/moments/StopGradientStopGradientGaudio_diffusion_conditioner/layer_normalization_2/moments/mean:output:0*
T0*'
_output_shapes
:����������
Kaudio_diffusion_conditioner/layer_normalization_2/moments/SquaredDifferenceSquaredDifference#audio_diffusion_conditioner/add:z:0Oaudio_diffusion_conditioner/layer_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Taudio_diffusion_conditioner/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
Baudio_diffusion_conditioner/layer_normalization_2/moments/varianceMeanOaudio_diffusion_conditioner/layer_normalization_2/moments/SquaredDifference:z:0]audio_diffusion_conditioner/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(�
Aaudio_diffusion_conditioner/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5�
?audio_diffusion_conditioner/layer_normalization_2/batchnorm/addAddV2Kaudio_diffusion_conditioner/layer_normalization_2/moments/variance:output:0Jaudio_diffusion_conditioner/layer_normalization_2/batchnorm/add/y:output:0*
T0*'
_output_shapes
:����������
Aaudio_diffusion_conditioner/layer_normalization_2/batchnorm/RsqrtRsqrtCaudio_diffusion_conditioner/layer_normalization_2/batchnorm/add:z:0*
T0*'
_output_shapes
:����������
Naudio_diffusion_conditioner/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpWaudio_diffusion_conditioner_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?audio_diffusion_conditioner/layer_normalization_2/batchnorm/mulMulEaudio_diffusion_conditioner/layer_normalization_2/batchnorm/Rsqrt:y:0Vaudio_diffusion_conditioner/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Aaudio_diffusion_conditioner/layer_normalization_2/batchnorm/mul_1Mul#audio_diffusion_conditioner/add:z:0Caudio_diffusion_conditioner/layer_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Aaudio_diffusion_conditioner/layer_normalization_2/batchnorm/mul_2MulGaudio_diffusion_conditioner/layer_normalization_2/moments/mean:output:0Caudio_diffusion_conditioner/layer_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
Jaudio_diffusion_conditioner/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpSaudio_diffusion_conditioner_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?audio_diffusion_conditioner/layer_normalization_2/batchnorm/subSubRaudio_diffusion_conditioner/layer_normalization_2/batchnorm/ReadVariableOp:value:0Eaudio_diffusion_conditioner/layer_normalization_2/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:�����������
Aaudio_diffusion_conditioner/layer_normalization_2/batchnorm/add_1AddV2Eaudio_diffusion_conditioner/layer_normalization_2/batchnorm/mul_1:z:0Caudio_diffusion_conditioner/layer_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
9audio_diffusion_conditioner/dense_8/MatMul/ReadVariableOpReadVariableOpBaudio_diffusion_conditioner_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
*audio_diffusion_conditioner/dense_8/MatMulMatMulEaudio_diffusion_conditioner/layer_normalization_2/batchnorm/add_1:z:0Aaudio_diffusion_conditioner/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:audio_diffusion_conditioner/dense_8/BiasAdd/ReadVariableOpReadVariableOpCaudio_diffusion_conditioner_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+audio_diffusion_conditioner/dense_8/BiasAddBiasAdd4audio_diffusion_conditioner/dense_8/MatMul:product:0Baudio_diffusion_conditioner/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(audio_diffusion_conditioner/dense_8/TanhTanh4audio_diffusion_conditioner/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
9audio_diffusion_conditioner/dense_9/MatMul/ReadVariableOpReadVariableOpBaudio_diffusion_conditioner_dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
*audio_diffusion_conditioner/dense_9/MatMulMatMulEaudio_diffusion_conditioner/layer_normalization_2/batchnorm/add_1:z:0Aaudio_diffusion_conditioner/dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:audio_diffusion_conditioner/dense_9/BiasAdd/ReadVariableOpReadVariableOpCaudio_diffusion_conditioner_dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+audio_diffusion_conditioner/dense_9/BiasAddBiasAdd4audio_diffusion_conditioner/dense_9/MatMul:product:0Baudio_diffusion_conditioner/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+audio_diffusion_conditioner/dense_9/SigmoidSigmoid4audio_diffusion_conditioner/dense_9/BiasAdd:output:0*
T0*(
_output_shapes
:����������r
'audio_diffusion_conditioner/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
"audio_diffusion_conditioner/concatConcatV2,audio_diffusion_conditioner/dense_8/Tanh:y:0/audio_diffusion_conditioner/dense_9/Sigmoid:y:00audio_diffusion_conditioner/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	{
IdentityIdentity+audio_diffusion_conditioner/concat:output:0^NoOp*
T0*(
_output_shapes
:����������	�
NoOpNoOp;^audio_diffusion_conditioner/dense_8/BiasAdd/ReadVariableOp:^audio_diffusion_conditioner/dense_8/MatMul/ReadVariableOp;^audio_diffusion_conditioner/dense_9/BiasAdd/ReadVariableOp:^audio_diffusion_conditioner/dense_9/MatMul/ReadVariableOpK^audio_diffusion_conditioner/layer_normalization_2/batchnorm/ReadVariableOpO^audio_diffusion_conditioner/layer_normalization_2/batchnorm/mul/ReadVariableOpW^audio_diffusion_conditioner/multi_head_attention_1/attention_output/add/ReadVariableOpa^audio_diffusion_conditioner/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpJ^audio_diffusion_conditioner/multi_head_attention_1/key/add/ReadVariableOpT^audio_diffusion_conditioner/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpL^audio_diffusion_conditioner/multi_head_attention_1/query/add/ReadVariableOpV^audio_diffusion_conditioner/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpL^audio_diffusion_conditioner/multi_head_attention_1/value/add/ReadVariableOpV^audio_diffusion_conditioner/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpF^audio_diffusion_conditioner/sequential/dense_5/BiasAdd/ReadVariableOpE^audio_diffusion_conditioner/sequential/dense_5/MatMul/ReadVariableOpF^audio_diffusion_conditioner/sequential/dense_6/BiasAdd/ReadVariableOpE^audio_diffusion_conditioner/sequential/dense_6/MatMul/ReadVariableOpF^audio_diffusion_conditioner/sequential/dense_7/BiasAdd/ReadVariableOpE^audio_diffusion_conditioner/sequential/dense_7/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : : : 2x
:audio_diffusion_conditioner/dense_8/BiasAdd/ReadVariableOp:audio_diffusion_conditioner/dense_8/BiasAdd/ReadVariableOp2v
9audio_diffusion_conditioner/dense_8/MatMul/ReadVariableOp9audio_diffusion_conditioner/dense_8/MatMul/ReadVariableOp2x
:audio_diffusion_conditioner/dense_9/BiasAdd/ReadVariableOp:audio_diffusion_conditioner/dense_9/BiasAdd/ReadVariableOp2v
9audio_diffusion_conditioner/dense_9/MatMul/ReadVariableOp9audio_diffusion_conditioner/dense_9/MatMul/ReadVariableOp2�
Jaudio_diffusion_conditioner/layer_normalization_2/batchnorm/ReadVariableOpJaudio_diffusion_conditioner/layer_normalization_2/batchnorm/ReadVariableOp2�
Naudio_diffusion_conditioner/layer_normalization_2/batchnorm/mul/ReadVariableOpNaudio_diffusion_conditioner/layer_normalization_2/batchnorm/mul/ReadVariableOp2�
Vaudio_diffusion_conditioner/multi_head_attention_1/attention_output/add/ReadVariableOpVaudio_diffusion_conditioner/multi_head_attention_1/attention_output/add/ReadVariableOp2�
`audio_diffusion_conditioner/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp`audio_diffusion_conditioner/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2�
Iaudio_diffusion_conditioner/multi_head_attention_1/key/add/ReadVariableOpIaudio_diffusion_conditioner/multi_head_attention_1/key/add/ReadVariableOp2�
Saudio_diffusion_conditioner/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpSaudio_diffusion_conditioner/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2�
Kaudio_diffusion_conditioner/multi_head_attention_1/query/add/ReadVariableOpKaudio_diffusion_conditioner/multi_head_attention_1/query/add/ReadVariableOp2�
Uaudio_diffusion_conditioner/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpUaudio_diffusion_conditioner/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2�
Kaudio_diffusion_conditioner/multi_head_attention_1/value/add/ReadVariableOpKaudio_diffusion_conditioner/multi_head_attention_1/value/add/ReadVariableOp2�
Uaudio_diffusion_conditioner/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpUaudio_diffusion_conditioner/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2�
Eaudio_diffusion_conditioner/sequential/dense_5/BiasAdd/ReadVariableOpEaudio_diffusion_conditioner/sequential/dense_5/BiasAdd/ReadVariableOp2�
Daudio_diffusion_conditioner/sequential/dense_5/MatMul/ReadVariableOpDaudio_diffusion_conditioner/sequential/dense_5/MatMul/ReadVariableOp2�
Eaudio_diffusion_conditioner/sequential/dense_6/BiasAdd/ReadVariableOpEaudio_diffusion_conditioner/sequential/dense_6/BiasAdd/ReadVariableOp2�
Daudio_diffusion_conditioner/sequential/dense_6/MatMul/ReadVariableOpDaudio_diffusion_conditioner/sequential/dense_6/MatMul/ReadVariableOp2�
Eaudio_diffusion_conditioner/sequential/dense_7/BiasAdd/ReadVariableOpEaudio_diffusion_conditioner/sequential/dense_7/BiasAdd/ReadVariableOp2�
Daudio_diffusion_conditioner/sequential/dense_7/MatMul/ReadVariableOpDaudio_diffusion_conditioner/sequential/dense_7/MatMul/ReadVariableOp:($
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
�
�
F__inference_sequential_layer_call_and_return_conditional_losses_225232
input_1"
dense_5_225204:
��
dense_5_225206:	�"
dense_6_225215:
��
dense_6_225217:	�"
dense_7_225226:
��
dense_7_225228:	�
identity��dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_5_225204dense_5_225206*
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
C__inference_dense_5_layer_call_and_return_conditional_losses_225130�
dropout_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_225213�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_6_225215dense_6_225217*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_225166�
dropout_3/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_225224�
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_7_225226dense_7_225228*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_225194x
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:&"
 
_user_specified_name225228:&"
 
_user_specified_name225226:&"
 
_user_specified_name225217:&"
 
_user_specified_name225215:&"
 
_user_specified_name225206:&"
 
_user_specified_name225204:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
<__inference_audio_diffusion_conditioner_layer_call_fn_225584
input_1
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	� 
	unknown_5:� 
	unknown_6:  
	unknown_7:� 
	unknown_8:  
	unknown_9:� 

unknown_10: !

unknown_11: �

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:
��

unknown_18:	�
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
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_audio_diffusion_conditioner_layer_call_and_return_conditional_losses_225448p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������	<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name225580:&"
 
_user_specified_name225578:&"
 
_user_specified_name225576:&"
 
_user_specified_name225574:&"
 
_user_specified_name225572:&"
 
_user_specified_name225570:&"
 
_user_specified_name225568:&"
 
_user_specified_name225566:&"
 
_user_specified_name225564:&"
 
_user_specified_name225562:&
"
 
_user_specified_name225560:&	"
 
_user_specified_name225558:&"
 
_user_specified_name225556:&"
 
_user_specified_name225554:&"
 
_user_specified_name225552:&"
 
_user_specified_name225550:&"
 
_user_specified_name225548:&"
 
_user_specified_name225546:&"
 
_user_specified_name225544:&"
 
_user_specified_name225542:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
C__inference_dense_6_layer_call_and_return_conditional_losses_226020

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:����������T
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:����������O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:����������`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:����������^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:����������S
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
C__inference_dense_5_layer_call_and_return_conditional_losses_225966

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
:����������O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:����������T
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:����������O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:����������`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:����������^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:����������S
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
C__inference_dense_6_layer_call_and_return_conditional_losses_225166

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:����������T
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:����������O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:����������`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:����������^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:����������S
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
�

�
C__inference_dense_9_layer_call_and_return_conditional_losses_225439

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
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
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_8_layer_call_fn_225877

inputs
unknown:
��
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
C__inference_dense_8_layer_call_and_return_conditional_losses_225423p
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
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name225873:&"
 
_user_specified_name225871:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_3_layer_call_and_return_conditional_losses_226042

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
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_225993

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�0
�	
W__inference_audio_diffusion_conditioner_layer_call_and_return_conditional_losses_225539
input_1%
sequential_225451:
�� 
sequential_225453:	�%
sequential_225455:
�� 
sequential_225457:	�%
sequential_225459:
�� 
sequential_225461:	�4
multi_head_attention_1_225502:� /
multi_head_attention_1_225504: 4
multi_head_attention_1_225506:� /
multi_head_attention_1_225508: 4
multi_head_attention_1_225510:� /
multi_head_attention_1_225512: 4
multi_head_attention_1_225514: �,
multi_head_attention_1_225516:	�+
layer_normalization_2_225521:	�+
layer_normalization_2_225523:	�"
dense_8_225526:
��
dense_8_225528:	�"
dense_9_225531:
��
dense_9_225533:	�
identity��dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�.multi_head_attention_1/StatefulPartitionedCall�"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_225451sequential_225453sequential_225455sequential_225457sequential_225459sequential_225461*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_225232P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�

ExpandDims
ExpandDims+sequential/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:�����������
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0ExpandDims:output:0ExpandDims:output:0multi_head_attention_1_225502multi_head_attention_1_225504multi_head_attention_1_225506multi_head_attention_1_225508multi_head_attention_1_225510multi_head_attention_1_225512multi_head_attention_1_225514multi_head_attention_1_225516*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225501�
SqueezeSqueeze7multi_head_attention_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������*
squeeze_dims
~
addAddV2+sequential/StatefulPartitionedCall:output:0Squeeze:output:0*
T0*(
_output_shapes
:�����������
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd:z:0layer_normalization_2_225521layer_normalization_2_225523*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_225407�
dense_8/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_8_225526dense_8_225528*
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
C__inference_dense_8_layer_call_and_return_conditional_losses_225423�
dense_9/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_9_225531dense_9_225533*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_225439V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
concatConcatV2(dense_8/StatefulPartitionedCall:output:0(dense_9/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	_
IdentityIdentityconcat:output:0^NoOp*
T0*(
_output_shapes
:����������	�
NoOpNoOp ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:&"
 
_user_specified_name225533:&"
 
_user_specified_name225531:&"
 
_user_specified_name225528:&"
 
_user_specified_name225526:&"
 
_user_specified_name225523:&"
 
_user_specified_name225521:&"
 
_user_specified_name225516:&"
 
_user_specified_name225514:&"
 
_user_specified_name225512:&"
 
_user_specified_name225510:&
"
 
_user_specified_name225508:&	"
 
_user_specified_name225506:&"
 
_user_specified_name225504:&"
 
_user_specified_name225502:&"
 
_user_specified_name225461:&"
 
_user_specified_name225459:&"
 
_user_specified_name225457:&"
 
_user_specified_name225455:&"
 
_user_specified_name225453:&"
 
_user_specified_name225451:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
��
�5
"__inference__traced_restore_226673
file_prefix3
assignvariableop_dense_5_kernel:
��.
assignvariableop_1_dense_5_bias:	�5
!assignvariableop_2_dense_6_kernel:
��.
assignvariableop_3_dense_6_bias:	�5
!assignvariableop_4_dense_7_kernel:
��.
assignvariableop_5_dense_7_bias:	�i
Rassignvariableop_6_audio_diffusion_conditioner_multi_head_attention_1_query_kernel:� b
Passignvariableop_7_audio_diffusion_conditioner_multi_head_attention_1_query_bias: g
Passignvariableop_8_audio_diffusion_conditioner_multi_head_attention_1_key_kernel:� `
Nassignvariableop_9_audio_diffusion_conditioner_multi_head_attention_1_key_bias: j
Sassignvariableop_10_audio_diffusion_conditioner_multi_head_attention_1_value_kernel:� c
Qassignvariableop_11_audio_diffusion_conditioner_multi_head_attention_1_value_bias: u
^assignvariableop_12_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel: �k
\assignvariableop_13_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias:	�R
>assignvariableop_14_audio_diffusion_conditioner_dense_8_kernel:
��K
<assignvariableop_15_audio_diffusion_conditioner_dense_8_bias:	�R
>assignvariableop_16_audio_diffusion_conditioner_dense_9_kernel:
��K
<assignvariableop_17_audio_diffusion_conditioner_dense_9_bias:	�Z
Kassignvariableop_18_audio_diffusion_conditioner_layer_normalization_2_gamma:	�Y
Jassignvariableop_19_audio_diffusion_conditioner_layer_normalization_2_beta:	�'
assignvariableop_20_iteration:	 +
!assignvariableop_21_learning_rate: =
)assignvariableop_22_adam_m_dense_5_kernel:
��=
)assignvariableop_23_adam_v_dense_5_kernel:
��6
'assignvariableop_24_adam_m_dense_5_bias:	�6
'assignvariableop_25_adam_v_dense_5_bias:	�=
)assignvariableop_26_adam_m_dense_6_kernel:
��=
)assignvariableop_27_adam_v_dense_6_kernel:
��6
'assignvariableop_28_adam_m_dense_6_bias:	�6
'assignvariableop_29_adam_v_dense_6_bias:	�=
)assignvariableop_30_adam_m_dense_7_kernel:
��=
)assignvariableop_31_adam_v_dense_7_kernel:
��6
'assignvariableop_32_adam_m_dense_7_bias:	�6
'assignvariableop_33_adam_v_dense_7_bias:	�q
Zassignvariableop_34_adam_m_audio_diffusion_conditioner_multi_head_attention_1_query_kernel:� q
Zassignvariableop_35_adam_v_audio_diffusion_conditioner_multi_head_attention_1_query_kernel:� j
Xassignvariableop_36_adam_m_audio_diffusion_conditioner_multi_head_attention_1_query_bias: j
Xassignvariableop_37_adam_v_audio_diffusion_conditioner_multi_head_attention_1_query_bias: o
Xassignvariableop_38_adam_m_audio_diffusion_conditioner_multi_head_attention_1_key_kernel:� o
Xassignvariableop_39_adam_v_audio_diffusion_conditioner_multi_head_attention_1_key_kernel:� h
Vassignvariableop_40_adam_m_audio_diffusion_conditioner_multi_head_attention_1_key_bias: h
Vassignvariableop_41_adam_v_audio_diffusion_conditioner_multi_head_attention_1_key_bias: q
Zassignvariableop_42_adam_m_audio_diffusion_conditioner_multi_head_attention_1_value_kernel:� q
Zassignvariableop_43_adam_v_audio_diffusion_conditioner_multi_head_attention_1_value_kernel:� j
Xassignvariableop_44_adam_m_audio_diffusion_conditioner_multi_head_attention_1_value_bias: j
Xassignvariableop_45_adam_v_audio_diffusion_conditioner_multi_head_attention_1_value_bias: |
eassignvariableop_46_adam_m_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel: �|
eassignvariableop_47_adam_v_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel: �r
cassignvariableop_48_adam_m_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias:	�r
cassignvariableop_49_adam_v_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias:	�Y
Eassignvariableop_50_adam_m_audio_diffusion_conditioner_dense_8_kernel:
��Y
Eassignvariableop_51_adam_v_audio_diffusion_conditioner_dense_8_kernel:
��R
Cassignvariableop_52_adam_m_audio_diffusion_conditioner_dense_8_bias:	�R
Cassignvariableop_53_adam_v_audio_diffusion_conditioner_dense_8_bias:	�Y
Eassignvariableop_54_adam_m_audio_diffusion_conditioner_dense_9_kernel:
��Y
Eassignvariableop_55_adam_v_audio_diffusion_conditioner_dense_9_kernel:
��R
Cassignvariableop_56_adam_m_audio_diffusion_conditioner_dense_9_bias:	�R
Cassignvariableop_57_adam_v_audio_diffusion_conditioner_dense_9_bias:	�a
Rassignvariableop_58_adam_m_audio_diffusion_conditioner_layer_normalization_2_gamma:	�a
Rassignvariableop_59_adam_v_audio_diffusion_conditioner_layer_normalization_2_gamma:	�`
Qassignvariableop_60_adam_m_audio_diffusion_conditioner_layer_normalization_2_beta:	�`
Qassignvariableop_61_adam_v_audio_diffusion_conditioner_layer_normalization_2_beta:	�#
assignvariableop_62_total: #
assignvariableop_63_count: 
identity_65��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*O
dtypesE
C2A	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_5_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_6_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_6_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_7_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_7_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpRassignvariableop_6_audio_diffusion_conditioner_multi_head_attention_1_query_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpPassignvariableop_7_audio_diffusion_conditioner_multi_head_attention_1_query_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpPassignvariableop_8_audio_diffusion_conditioner_multi_head_attention_1_key_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpNassignvariableop_9_audio_diffusion_conditioner_multi_head_attention_1_key_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpSassignvariableop_10_audio_diffusion_conditioner_multi_head_attention_1_value_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpQassignvariableop_11_audio_diffusion_conditioner_multi_head_attention_1_value_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp^assignvariableop_12_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp\assignvariableop_13_audio_diffusion_conditioner_multi_head_attention_1_attention_output_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp>assignvariableop_14_audio_diffusion_conditioner_dense_8_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp<assignvariableop_15_audio_diffusion_conditioner_dense_8_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp>assignvariableop_16_audio_diffusion_conditioner_dense_9_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp<assignvariableop_17_audio_diffusion_conditioner_dense_9_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpKassignvariableop_18_audio_diffusion_conditioner_layer_normalization_2_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpJassignvariableop_19_audio_diffusion_conditioner_layer_normalization_2_betaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_iterationIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_learning_rateIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_m_dense_5_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_v_dense_5_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_m_dense_5_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_v_dense_5_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_m_dense_6_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_v_dense_6_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_m_dense_6_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_v_dense_6_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_m_dense_7_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_v_dense_7_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_m_dense_7_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_v_dense_7_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpZassignvariableop_34_adam_m_audio_diffusion_conditioner_multi_head_attention_1_query_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpZassignvariableop_35_adam_v_audio_diffusion_conditioner_multi_head_attention_1_query_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpXassignvariableop_36_adam_m_audio_diffusion_conditioner_multi_head_attention_1_query_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpXassignvariableop_37_adam_v_audio_diffusion_conditioner_multi_head_attention_1_query_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpXassignvariableop_38_adam_m_audio_diffusion_conditioner_multi_head_attention_1_key_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpXassignvariableop_39_adam_v_audio_diffusion_conditioner_multi_head_attention_1_key_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpVassignvariableop_40_adam_m_audio_diffusion_conditioner_multi_head_attention_1_key_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpVassignvariableop_41_adam_v_audio_diffusion_conditioner_multi_head_attention_1_key_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpZassignvariableop_42_adam_m_audio_diffusion_conditioner_multi_head_attention_1_value_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpZassignvariableop_43_adam_v_audio_diffusion_conditioner_multi_head_attention_1_value_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpXassignvariableop_44_adam_m_audio_diffusion_conditioner_multi_head_attention_1_value_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpXassignvariableop_45_adam_v_audio_diffusion_conditioner_multi_head_attention_1_value_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpeassignvariableop_46_adam_m_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpeassignvariableop_47_adam_v_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpcassignvariableop_48_adam_m_audio_diffusion_conditioner_multi_head_attention_1_attention_output_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpcassignvariableop_49_adam_v_audio_diffusion_conditioner_multi_head_attention_1_attention_output_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpEassignvariableop_50_adam_m_audio_diffusion_conditioner_dense_8_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpEassignvariableop_51_adam_v_audio_diffusion_conditioner_dense_8_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpCassignvariableop_52_adam_m_audio_diffusion_conditioner_dense_8_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpCassignvariableop_53_adam_v_audio_diffusion_conditioner_dense_8_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpEassignvariableop_54_adam_m_audio_diffusion_conditioner_dense_9_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpEassignvariableop_55_adam_v_audio_diffusion_conditioner_dense_9_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpCassignvariableop_56_adam_m_audio_diffusion_conditioner_dense_9_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpCassignvariableop_57_adam_v_audio_diffusion_conditioner_dense_9_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpRassignvariableop_58_adam_m_audio_diffusion_conditioner_layer_normalization_2_gammaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpRassignvariableop_59_adam_v_audio_diffusion_conditioner_layer_normalization_2_gammaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpQassignvariableop_60_adam_m_audio_diffusion_conditioner_layer_normalization_2_betaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpQassignvariableop_61_adam_v_audio_diffusion_conditioner_layer_normalization_2_betaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpassignvariableop_62_totalIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpassignvariableop_63_countIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_64Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_65IdentityIdentity_64:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_65Identity_65:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_63AssignVariableOp_632(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%@!

_user_specified_namecount:%?!

_user_specified_nametotal:]>Y
W
_user_specified_name?=Adam/v/audio_diffusion_conditioner/layer_normalization_2/beta:]=Y
W
_user_specified_name?=Adam/m/audio_diffusion_conditioner/layer_normalization_2/beta:^<Z
X
_user_specified_name@>Adam/v/audio_diffusion_conditioner/layer_normalization_2/gamma:^;Z
X
_user_specified_name@>Adam/m/audio_diffusion_conditioner/layer_normalization_2/gamma:O:K
I
_user_specified_name1/Adam/v/audio_diffusion_conditioner/dense_9/bias:O9K
I
_user_specified_name1/Adam/m/audio_diffusion_conditioner/dense_9/bias:Q8M
K
_user_specified_name31Adam/v/audio_diffusion_conditioner/dense_9/kernel:Q7M
K
_user_specified_name31Adam/m/audio_diffusion_conditioner/dense_9/kernel:O6K
I
_user_specified_name1/Adam/v/audio_diffusion_conditioner/dense_8/bias:O5K
I
_user_specified_name1/Adam/m/audio_diffusion_conditioner/dense_8/bias:Q4M
K
_user_specified_name31Adam/v/audio_diffusion_conditioner/dense_8/kernel:Q3M
K
_user_specified_name31Adam/m/audio_diffusion_conditioner/dense_8/kernel:o2k
i
_user_specified_nameQOAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias:o1k
i
_user_specified_nameQOAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias:q0m
k
_user_specified_nameSQAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel:q/m
k
_user_specified_nameSQAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel:d.`
^
_user_specified_nameFDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/bias:d-`
^
_user_specified_nameFDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/bias:f,b
`
_user_specified_nameHFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/kernel:f+b
`
_user_specified_nameHFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/kernel:b*^
\
_user_specified_nameDBAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/bias:b)^
\
_user_specified_nameDBAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/bias:d(`
^
_user_specified_nameFDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/kernel:d'`
^
_user_specified_nameFDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/kernel:d&`
^
_user_specified_nameFDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/bias:d%`
^
_user_specified_nameFDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/bias:f$b
`
_user_specified_nameHFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/kernel:f#b
`
_user_specified_nameHFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/kernel:3"/
-
_user_specified_nameAdam/v/dense_7/bias:3!/
-
_user_specified_nameAdam/m/dense_7/bias:5 1
/
_user_specified_nameAdam/v/dense_7/kernel:51
/
_user_specified_nameAdam/m/dense_7/kernel:3/
-
_user_specified_nameAdam/v/dense_6/bias:3/
-
_user_specified_nameAdam/m/dense_6/bias:51
/
_user_specified_nameAdam/v/dense_6/kernel:51
/
_user_specified_nameAdam/m/dense_6/kernel:3/
-
_user_specified_nameAdam/v/dense_5/bias:3/
-
_user_specified_nameAdam/m/dense_5/bias:51
/
_user_specified_nameAdam/v/dense_5/kernel:51
/
_user_specified_nameAdam/m/dense_5/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:VR
P
_user_specified_name86audio_diffusion_conditioner/layer_normalization_2/beta:WS
Q
_user_specified_name97audio_diffusion_conditioner/layer_normalization_2/gamma:HD
B
_user_specified_name*(audio_diffusion_conditioner/dense_9/bias:JF
D
_user_specified_name,*audio_diffusion_conditioner/dense_9/kernel:HD
B
_user_specified_name*(audio_diffusion_conditioner/dense_8/bias:JF
D
_user_specified_name,*audio_diffusion_conditioner/dense_8/kernel:hd
b
_user_specified_nameJHaudio_diffusion_conditioner/multi_head_attention_1/attention_output/bias:jf
d
_user_specified_nameLJaudio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel:]Y
W
_user_specified_name?=audio_diffusion_conditioner/multi_head_attention_1/value/bias:_[
Y
_user_specified_nameA?audio_diffusion_conditioner/multi_head_attention_1/value/kernel:[
W
U
_user_specified_name=;audio_diffusion_conditioner/multi_head_attention_1/key/bias:]	Y
W
_user_specified_name?=audio_diffusion_conditioner/multi_head_attention_1/key/kernel:]Y
W
_user_specified_name?=audio_diffusion_conditioner/multi_head_attention_1/query/bias:_[
Y
_user_specified_nameA?audio_diffusion_conditioner/multi_head_attention_1/query/kernel:,(
&
_user_specified_namedense_7/bias:.*
(
_user_specified_namedense_7/kernel:,(
&
_user_specified_namedense_6/bias:.*
(
_user_specified_namedense_6/kernel:,(
&
_user_specified_namedense_5/bias:.*
(
_user_specified_namedense_5/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_225224

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
*__inference_dropout_3_layer_call_fn_226025

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_225183p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_7_layer_call_and_return_conditional_losses_226066

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
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
:����������
 
_user_specified_nameinputs
�-
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225832	
query	
value
keyB
+query_einsum_einsum_readvariableop_resource:� 3
!query_add_readvariableop_resource: @
)key_einsum_einsum_readvariableop_resource:� 1
key_add_readvariableop_resource: B
+value_einsum_einsum_readvariableop_resource:� 3
!value_add_readvariableop_resource: M
6attention_output_einsum_einsum_readvariableop_resource: �;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:��������� �
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
:��������� *
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 2J
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
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery
�
�
(__inference_dense_6_layer_call_fn_226002

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_225166p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name225998:&"
 
_user_specified_name225996:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
+__inference_sequential_layer_call_fn_225266
input_1
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_225232p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name225262:&"
 
_user_specified_name225260:&"
 
_user_specified_name225258:&"
 
_user_specified_name225256:&"
 
_user_specified_name225254:&"
 
_user_specified_name225252:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�-
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225868	
query	
value
keyB
+query_einsum_einsum_readvariableop_resource:� 3
!query_add_readvariableop_resource: @
)key_einsum_einsum_readvariableop_resource:� 1
key_add_readvariableop_resource: B
+value_einsum_einsum_readvariableop_resource:� 3
!value_add_readvariableop_resource: M
6attention_output_einsum_einsum_readvariableop_resource: �;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:��������� �
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
:��������� *
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 2J
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
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery
�
�
$__inference_signature_wrapper_225750
input_1
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	� 
	unknown_5:� 
	unknown_6:  
	unknown_7:� 
	unknown_8:  
	unknown_9:� 

unknown_10: !

unknown_11: �

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:
��

unknown_18:	�
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
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_225110p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������	<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name225746:&"
 
_user_specified_name225744:&"
 
_user_specified_name225742:&"
 
_user_specified_name225740:&"
 
_user_specified_name225738:&"
 
_user_specified_name225736:&"
 
_user_specified_name225734:&"
 
_user_specified_name225732:&"
 
_user_specified_name225730:&"
 
_user_specified_name225728:&
"
 
_user_specified_name225726:&	"
 
_user_specified_name225724:&"
 
_user_specified_name225722:&"
 
_user_specified_name225720:&"
 
_user_specified_name225718:&"
 
_user_specified_name225716:&"
 
_user_specified_name225714:&"
 
_user_specified_name225712:&"
 
_user_specified_name225710:&"
 
_user_specified_name225708:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
(__inference_dense_5_layer_call_fn_225948

inputs
unknown:
��
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
C__inference_dense_5_layer_call_and_return_conditional_losses_225130p
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
 
_user_specified_name225944:&"
 
_user_specified_name225942:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_225988

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
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_225213

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_2_layer_call_and_return_conditional_losses_225147

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
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�-
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225501	
query	
value
keyB
+query_einsum_einsum_readvariableop_resource:� 3
!query_add_readvariableop_resource: @
)key_einsum_einsum_readvariableop_resource:� 1
key_add_readvariableop_resource: B
+value_einsum_einsum_readvariableop_resource:� 3
!value_add_readvariableop_resource: M
6attention_output_einsum_einsum_readvariableop_resource: �;
,attention_output_add_readvariableop_resource:	�
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

: *
dtype0�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*#
_output_shapes
:� *
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:��������� *
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

: *
dtype0�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:��������� �
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
:��������� *
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*#
_output_shapes
: �*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������l
IdentityIdentityattention_output/add:z:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 2J
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
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery
�
�
(__inference_dense_7_layer_call_fn_226056

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_225194p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name226052:&"
 
_user_specified_name226050:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_9_layer_call_fn_225897

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_225439p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name225893:&"
 
_user_specified_name225891:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_layer_normalization_2_layer_call_fn_225917

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_225407p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name225913:&"
 
_user_specified_name225911:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_8_layer_call_and_return_conditional_losses_225423

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
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
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_5_layer_call_and_return_conditional_losses_225130

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
:����������O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?i
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?r
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*(
_output_shapes
:����������T
Gelu/ErfErfGelu/truediv:z:0*
T0*(
_output_shapes
:����������O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*(
_output_shapes
:����������`

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*(
_output_shapes
:����������^
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*(
_output_shapes
:����������S
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
�
�
F__inference_sequential_layer_call_and_return_conditional_losses_225201
input_1"
dense_5_225131:
��
dense_5_225133:	�"
dense_6_225167:
��
dense_6_225169:	�"
dense_7_225195:
��
dense_7_225197:	�
identity��dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_5_225131dense_5_225133*
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
C__inference_dense_5_layer_call_and_return_conditional_losses_225130�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_225147�
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_6_225167dense_6_225169*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_225166�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_225183�
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_7_225195dense_7_225197*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_225194x
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:&"
 
_user_specified_name225197:&"
 
_user_specified_name225195:&"
 
_user_specified_name225169:&"
 
_user_specified_name225167:&"
 
_user_specified_name225133:&"
 
_user_specified_name225131:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�

�
C__inference_dense_9_layer_call_and_return_conditional_losses_225908

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
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
:����������
 
_user_specified_nameinputs
�
�
7__inference_multi_head_attention_1_layer_call_fn_225796	
query	
value
key
unknown:� 
	unknown_0:  
	unknown_1:� 
	unknown_2:  
	unknown_3:� 
	unknown_4:  
	unknown_5: �
	unknown_6:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225501t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&
"
 
_user_specified_name225792:&	"
 
_user_specified_name225790:&"
 
_user_specified_name225788:&"
 
_user_specified_name225786:&"
 
_user_specified_name225784:&"
 
_user_specified_name225782:&"
 
_user_specified_name225780:&"
 
_user_specified_name225778:QM
,
_output_shapes
:����������

_user_specified_namekey:SO
,
_output_shapes
:����������

_user_specified_namevalue:S O
,
_output_shapes
:����������

_user_specified_namequery
͗
�H
__inference__traced_save_226472
file_prefix9
%read_disablecopyonread_dense_5_kernel:
��4
%read_1_disablecopyonread_dense_5_bias:	�;
'read_2_disablecopyonread_dense_6_kernel:
��4
%read_3_disablecopyonread_dense_6_bias:	�;
'read_4_disablecopyonread_dense_7_kernel:
��4
%read_5_disablecopyonread_dense_7_bias:	�o
Xread_6_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_query_kernel:� h
Vread_7_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_query_bias: m
Vread_8_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_key_kernel:� f
Tread_9_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_key_bias: p
Yread_10_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_value_kernel:� i
Wread_11_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_value_bias: {
dread_12_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel: �q
bread_13_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias:	�X
Dread_14_disablecopyonread_audio_diffusion_conditioner_dense_8_kernel:
��Q
Bread_15_disablecopyonread_audio_diffusion_conditioner_dense_8_bias:	�X
Dread_16_disablecopyonread_audio_diffusion_conditioner_dense_9_kernel:
��Q
Bread_17_disablecopyonread_audio_diffusion_conditioner_dense_9_bias:	�`
Qread_18_disablecopyonread_audio_diffusion_conditioner_layer_normalization_2_gamma:	�_
Pread_19_disablecopyonread_audio_diffusion_conditioner_layer_normalization_2_beta:	�-
#read_20_disablecopyonread_iteration:	 1
'read_21_disablecopyonread_learning_rate: C
/read_22_disablecopyonread_adam_m_dense_5_kernel:
��C
/read_23_disablecopyonread_adam_v_dense_5_kernel:
��<
-read_24_disablecopyonread_adam_m_dense_5_bias:	�<
-read_25_disablecopyonread_adam_v_dense_5_bias:	�C
/read_26_disablecopyonread_adam_m_dense_6_kernel:
��C
/read_27_disablecopyonread_adam_v_dense_6_kernel:
��<
-read_28_disablecopyonread_adam_m_dense_6_bias:	�<
-read_29_disablecopyonread_adam_v_dense_6_bias:	�C
/read_30_disablecopyonread_adam_m_dense_7_kernel:
��C
/read_31_disablecopyonread_adam_v_dense_7_kernel:
��<
-read_32_disablecopyonread_adam_m_dense_7_bias:	�<
-read_33_disablecopyonread_adam_v_dense_7_bias:	�w
`read_34_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_query_kernel:� w
`read_35_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_query_kernel:� p
^read_36_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_query_bias: p
^read_37_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_query_bias: u
^read_38_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_key_kernel:� u
^read_39_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_key_kernel:� n
\read_40_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_key_bias: n
\read_41_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_key_bias: w
`read_42_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_value_kernel:� w
`read_43_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_value_kernel:� p
^read_44_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_value_bias: p
^read_45_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_value_bias: �
kread_46_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel: ��
kread_47_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel: �x
iread_48_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias:	�x
iread_49_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias:	�_
Kread_50_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_8_kernel:
��_
Kread_51_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_8_kernel:
��X
Iread_52_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_8_bias:	�X
Iread_53_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_8_bias:	�_
Kread_54_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_9_kernel:
��_
Kread_55_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_9_kernel:
��X
Iread_56_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_9_bias:	�X
Iread_57_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_9_bias:	�g
Xread_58_disablecopyonread_adam_m_audio_diffusion_conditioner_layer_normalization_2_gamma:	�g
Xread_59_disablecopyonread_adam_v_audio_diffusion_conditioner_layer_normalization_2_gamma:	�f
Wread_60_disablecopyonread_adam_m_audio_diffusion_conditioner_layer_normalization_2_beta:	�f
Wread_61_disablecopyonread_adam_v_audio_diffusion_conditioner_layer_normalization_2_beta:	�)
read_62_disablecopyonread_total: )
read_63_disablecopyonread_count: 
savev2_const
identity_129��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp%read_disablecopyonread_dense_5_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_dense_5_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_6_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_6_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_7_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_7_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnReadXread_6_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpXread_6_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_query_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0s
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_7/DisableCopyOnReadDisableCopyOnReadVread_7_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_query_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpVread_7_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_query_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_8/DisableCopyOnReadDisableCopyOnReadVread_8_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpVread_8_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_key_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0s
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_9/DisableCopyOnReadDisableCopyOnReadTread_9_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_key_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpTread_9_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_key_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_10/DisableCopyOnReadDisableCopyOnReadYread_10_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpYread_10_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_value_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0t
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_11/DisableCopyOnReadDisableCopyOnReadWread_11_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_value_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpWread_11_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_value_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_12/DisableCopyOnReadDisableCopyOnReaddread_12_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpdread_12_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
: �*
dtype0t
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
: �j
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*#
_output_shapes
: ��
Read_13/DisableCopyOnReadDisableCopyOnReadbread_13_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpbread_13_disablecopyonread_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_14/DisableCopyOnReadDisableCopyOnReadDread_14_disablecopyonread_audio_diffusion_conditioner_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpDread_14_disablecopyonread_audio_diffusion_conditioner_dense_8_kernel^Read_14/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_15/DisableCopyOnReadDisableCopyOnReadBread_15_disablecopyonread_audio_diffusion_conditioner_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpBread_15_disablecopyonread_audio_diffusion_conditioner_dense_8_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
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
Read_16/DisableCopyOnReadDisableCopyOnReadDread_16_disablecopyonread_audio_diffusion_conditioner_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpDread_16_disablecopyonread_audio_diffusion_conditioner_dense_9_kernel^Read_16/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_17/DisableCopyOnReadDisableCopyOnReadBread_17_disablecopyonread_audio_diffusion_conditioner_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpBread_17_disablecopyonread_audio_diffusion_conditioner_dense_9_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnReadQread_18_disablecopyonread_audio_diffusion_conditioner_layer_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpQread_18_disablecopyonread_audio_diffusion_conditioner_layer_normalization_2_gamma^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_19/DisableCopyOnReadDisableCopyOnReadPread_19_disablecopyonread_audio_diffusion_conditioner_layer_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpPread_19_disablecopyonread_audio_diffusion_conditioner_layer_normalization_2_beta^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:�x
Read_20/DisableCopyOnReadDisableCopyOnRead#read_20_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp#read_20_disablecopyonread_iteration^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_learning_rate^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_22/DisableCopyOnReadDisableCopyOnRead/read_22_disablecopyonread_adam_m_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp/read_22_disablecopyonread_adam_m_dense_5_kernel^Read_22/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_23/DisableCopyOnReadDisableCopyOnRead/read_23_disablecopyonread_adam_v_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp/read_23_disablecopyonread_adam_v_dense_5_kernel^Read_23/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_24/DisableCopyOnReadDisableCopyOnRead-read_24_disablecopyonread_adam_m_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp-read_24_disablecopyonread_adam_m_dense_5_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_25/DisableCopyOnReadDisableCopyOnRead-read_25_disablecopyonread_adam_v_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp-read_25_disablecopyonread_adam_v_dense_5_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_26/DisableCopyOnReadDisableCopyOnRead/read_26_disablecopyonread_adam_m_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp/read_26_disablecopyonread_adam_m_dense_6_kernel^Read_26/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_27/DisableCopyOnReadDisableCopyOnRead/read_27_disablecopyonread_adam_v_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp/read_27_disablecopyonread_adam_v_dense_6_kernel^Read_27/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_28/DisableCopyOnReadDisableCopyOnRead-read_28_disablecopyonread_adam_m_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp-read_28_disablecopyonread_adam_m_dense_6_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_29/DisableCopyOnReadDisableCopyOnRead-read_29_disablecopyonread_adam_v_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp-read_29_disablecopyonread_adam_v_dense_6_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_30/DisableCopyOnReadDisableCopyOnRead/read_30_disablecopyonread_adam_m_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp/read_30_disablecopyonread_adam_m_dense_7_kernel^Read_30/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_31/DisableCopyOnReadDisableCopyOnRead/read_31_disablecopyonread_adam_v_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp/read_31_disablecopyonread_adam_v_dense_7_kernel^Read_31/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_32/DisableCopyOnReadDisableCopyOnRead-read_32_disablecopyonread_adam_m_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp-read_32_disablecopyonread_adam_m_dense_7_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_33/DisableCopyOnReadDisableCopyOnRead-read_33_disablecopyonread_adam_v_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp-read_33_disablecopyonread_adam_v_dense_7_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_34/DisableCopyOnReadDisableCopyOnRead`read_34_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp`read_34_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_query_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0t
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_35/DisableCopyOnReadDisableCopyOnRead`read_35_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp`read_35_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_query_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0t
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_36/DisableCopyOnReadDisableCopyOnRead^read_36_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_query_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp^read_36_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_query_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_37/DisableCopyOnReadDisableCopyOnRead^read_37_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_query_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp^read_37_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_query_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_38/DisableCopyOnReadDisableCopyOnRead^read_38_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp^read_38_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_key_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0t
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_39/DisableCopyOnReadDisableCopyOnRead^read_39_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp^read_39_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_key_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0t
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_40/DisableCopyOnReadDisableCopyOnRead\read_40_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_key_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp\read_40_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_key_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_41/DisableCopyOnReadDisableCopyOnRead\read_41_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_key_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp\read_41_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_key_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_42/DisableCopyOnReadDisableCopyOnRead`read_42_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp`read_42_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_value_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0t
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_43/DisableCopyOnReadDisableCopyOnRead`read_43_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp`read_43_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_value_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
:� *
dtype0t
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
:� j
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*#
_output_shapes
:� �
Read_44/DisableCopyOnReadDisableCopyOnRead^read_44_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_value_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp^read_44_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_value_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_45/DisableCopyOnReadDisableCopyOnRead^read_45_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_value_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp^read_45_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_value_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_46/DisableCopyOnReadDisableCopyOnReadkread_46_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOpkread_46_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
: �*
dtype0t
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
: �j
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*#
_output_shapes
: ��
Read_47/DisableCopyOnReadDisableCopyOnReadkread_47_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOpkread_47_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_attention_output_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*#
_output_shapes
: �*
dtype0t
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*#
_output_shapes
: �j
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*#
_output_shapes
: ��
Read_48/DisableCopyOnReadDisableCopyOnReadiread_48_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOpiread_48_disablecopyonread_adam_m_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_49/DisableCopyOnReadDisableCopyOnReadiread_49_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOpiread_49_disablecopyonread_adam_v_audio_diffusion_conditioner_multi_head_attention_1_attention_output_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_50/DisableCopyOnReadDisableCopyOnReadKread_50_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOpKread_50_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_8_kernel^Read_50/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_51/DisableCopyOnReadDisableCopyOnReadKread_51_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOpKread_51_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_8_kernel^Read_51/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_52/DisableCopyOnReadDisableCopyOnReadIread_52_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpIread_52_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_8_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_53/DisableCopyOnReadDisableCopyOnReadIread_53_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpIread_53_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_8_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_54/DisableCopyOnReadDisableCopyOnReadKread_54_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpKread_54_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_9_kernel^Read_54/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_55/DisableCopyOnReadDisableCopyOnReadKread_55_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOpKread_55_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_9_kernel^Read_55/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0r
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_56/DisableCopyOnReadDisableCopyOnReadIread_56_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOpIread_56_disablecopyonread_adam_m_audio_diffusion_conditioner_dense_9_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_57/DisableCopyOnReadDisableCopyOnReadIread_57_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOpIread_57_disablecopyonread_adam_v_audio_diffusion_conditioner_dense_9_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_58/DisableCopyOnReadDisableCopyOnReadXread_58_disablecopyonread_adam_m_audio_diffusion_conditioner_layer_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOpXread_58_disablecopyonread_adam_m_audio_diffusion_conditioner_layer_normalization_2_gamma^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_59/DisableCopyOnReadDisableCopyOnReadXread_59_disablecopyonread_adam_v_audio_diffusion_conditioner_layer_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOpXread_59_disablecopyonread_adam_v_audio_diffusion_conditioner_layer_normalization_2_gamma^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_60/DisableCopyOnReadDisableCopyOnReadWread_60_disablecopyonread_adam_m_audio_diffusion_conditioner_layer_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOpWread_60_disablecopyonread_adam_m_audio_diffusion_conditioner_layer_normalization_2_beta^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_61/DisableCopyOnReadDisableCopyOnReadWread_61_disablecopyonread_adam_v_audio_diffusion_conditioner_layer_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOpWread_61_disablecopyonread_adam_v_audio_diffusion_conditioner_layer_normalization_2_beta^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes	
:�t
Read_62/DisableCopyOnReadDisableCopyOnReadread_62_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOpread_62_disablecopyonread_total^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_63/DisableCopyOnReadDisableCopyOnReadread_63_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOpread_63_disablecopyonread_count^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *O
dtypesE
C2A	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_128Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_129IdentityIdentity_128:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_129Identity_129:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_63/ReadVariableOpRead_63/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=A9

_output_shapes
: 

_user_specified_nameConst:%@!

_user_specified_namecount:%?!

_user_specified_nametotal:]>Y
W
_user_specified_name?=Adam/v/audio_diffusion_conditioner/layer_normalization_2/beta:]=Y
W
_user_specified_name?=Adam/m/audio_diffusion_conditioner/layer_normalization_2/beta:^<Z
X
_user_specified_name@>Adam/v/audio_diffusion_conditioner/layer_normalization_2/gamma:^;Z
X
_user_specified_name@>Adam/m/audio_diffusion_conditioner/layer_normalization_2/gamma:O:K
I
_user_specified_name1/Adam/v/audio_diffusion_conditioner/dense_9/bias:O9K
I
_user_specified_name1/Adam/m/audio_diffusion_conditioner/dense_9/bias:Q8M
K
_user_specified_name31Adam/v/audio_diffusion_conditioner/dense_9/kernel:Q7M
K
_user_specified_name31Adam/m/audio_diffusion_conditioner/dense_9/kernel:O6K
I
_user_specified_name1/Adam/v/audio_diffusion_conditioner/dense_8/bias:O5K
I
_user_specified_name1/Adam/m/audio_diffusion_conditioner/dense_8/bias:Q4M
K
_user_specified_name31Adam/v/audio_diffusion_conditioner/dense_8/kernel:Q3M
K
_user_specified_name31Adam/m/audio_diffusion_conditioner/dense_8/kernel:o2k
i
_user_specified_nameQOAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias:o1k
i
_user_specified_nameQOAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias:q0m
k
_user_specified_nameSQAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel:q/m
k
_user_specified_nameSQAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel:d.`
^
_user_specified_nameFDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/bias:d-`
^
_user_specified_nameFDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/bias:f,b
`
_user_specified_nameHFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/kernel:f+b
`
_user_specified_nameHFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/kernel:b*^
\
_user_specified_nameDBAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/bias:b)^
\
_user_specified_nameDBAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/bias:d(`
^
_user_specified_nameFDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/kernel:d'`
^
_user_specified_nameFDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/kernel:d&`
^
_user_specified_nameFDAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/bias:d%`
^
_user_specified_nameFDAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/bias:f$b
`
_user_specified_nameHFAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/kernel:f#b
`
_user_specified_nameHFAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/kernel:3"/
-
_user_specified_nameAdam/v/dense_7/bias:3!/
-
_user_specified_nameAdam/m/dense_7/bias:5 1
/
_user_specified_nameAdam/v/dense_7/kernel:51
/
_user_specified_nameAdam/m/dense_7/kernel:3/
-
_user_specified_nameAdam/v/dense_6/bias:3/
-
_user_specified_nameAdam/m/dense_6/bias:51
/
_user_specified_nameAdam/v/dense_6/kernel:51
/
_user_specified_nameAdam/m/dense_6/kernel:3/
-
_user_specified_nameAdam/v/dense_5/bias:3/
-
_user_specified_nameAdam/m/dense_5/bias:51
/
_user_specified_nameAdam/v/dense_5/kernel:51
/
_user_specified_nameAdam/m/dense_5/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:VR
P
_user_specified_name86audio_diffusion_conditioner/layer_normalization_2/beta:WS
Q
_user_specified_name97audio_diffusion_conditioner/layer_normalization_2/gamma:HD
B
_user_specified_name*(audio_diffusion_conditioner/dense_9/bias:JF
D
_user_specified_name,*audio_diffusion_conditioner/dense_9/kernel:HD
B
_user_specified_name*(audio_diffusion_conditioner/dense_8/bias:JF
D
_user_specified_name,*audio_diffusion_conditioner/dense_8/kernel:hd
b
_user_specified_nameJHaudio_diffusion_conditioner/multi_head_attention_1/attention_output/bias:jf
d
_user_specified_nameLJaudio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel:]Y
W
_user_specified_name?=audio_diffusion_conditioner/multi_head_attention_1/value/bias:_[
Y
_user_specified_nameA?audio_diffusion_conditioner/multi_head_attention_1/value/kernel:[
W
U
_user_specified_name=;audio_diffusion_conditioner/multi_head_attention_1/key/bias:]	Y
W
_user_specified_name?=audio_diffusion_conditioner/multi_head_attention_1/key/kernel:]Y
W
_user_specified_name?=audio_diffusion_conditioner/multi_head_attention_1/query/bias:_[
Y
_user_specified_nameA?audio_diffusion_conditioner/multi_head_attention_1/query/kernel:,(
&
_user_specified_namedense_7/bias:.*
(
_user_specified_namedense_7/kernel:,(
&
_user_specified_namedense_6/bias:.*
(
_user_specified_namedense_6/kernel:,(
&
_user_specified_namedense_5/bias:.*
(
_user_specified_namedense_5/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

d
E__inference_dropout_3_layer_call_and_return_conditional_losses_225183

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
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
*__inference_dropout_2_layer_call_fn_225971

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_225147p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_2_layer_call_fn_225976

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_225213a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
<__inference_audio_diffusion_conditioner_layer_call_fn_225629
input_1
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	� 
	unknown_5:� 
	unknown_6:  
	unknown_7:� 
	unknown_8:  
	unknown_9:� 

unknown_10: !

unknown_11: �

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:
��

unknown_18:	�
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
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_audio_diffusion_conditioner_layer_call_and_return_conditional_losses_225539p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������	<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:����������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name225625:&"
 
_user_specified_name225623:&"
 
_user_specified_name225621:&"
 
_user_specified_name225619:&"
 
_user_specified_name225617:&"
 
_user_specified_name225615:&"
 
_user_specified_name225613:&"
 
_user_specified_name225611:&"
 
_user_specified_name225609:&"
 
_user_specified_name225607:&
"
 
_user_specified_name225605:&	"
 
_user_specified_name225603:&"
 
_user_specified_name225601:&"
 
_user_specified_name225599:&"
 
_user_specified_name225597:&"
 
_user_specified_name225595:&"
 
_user_specified_name225593:&"
 
_user_specified_name225591:&"
 
_user_specified_name225589:&"
 
_user_specified_name225587:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1"�L
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
StatefulPartitionedCall:0����������	tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
text_encoder
		attention

frequency_proj
volume_proj
	layernorm
	optimizer

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19"
trackable_list_wrapper
 "
trackable_list_wrapper
�
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
(trace_0
)trace_12�
<__inference_audio_diffusion_conditioner_layer_call_fn_225584
<__inference_audio_diffusion_conditioner_layer_call_fn_225629�
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
 z(trace_0z)trace_1
�
*trace_0
+trace_12�
W__inference_audio_diffusion_conditioner_layer_call_and_return_conditional_losses_225448
W__inference_audio_diffusion_conditioner_layer_call_and_return_conditional_losses_225539�
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
 z*trace_0z+trace_1
�B�
!__inference__wrapped_model_225110input_1"�
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
�
,layer_with_weights-0
,layer-0
-layer-1
.layer_with_weights-1
.layer-2
/layer-3
0layer_with_weights-2
0layer-4
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=_query_dense
>
_key_dense
?_value_dense
@_softmax
A_dropout_layer
B_output_dense"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Uaxis
	!gamma
"beta"
_tf_keras_layer
�
V
_variables
W_iterations
X_learning_rate
Y_index_dict
Z
_momentums
[_velocities
\_update_step_xla"
experimentalOptimizer
,
]serving_default"
signature_map
": 
��2dense_5/kernel
:�2dense_5/bias
": 
��2dense_6/kernel
:�2dense_6/bias
": 
��2dense_7/kernel
:�2dense_7/bias
V:T� 2?audio_diffusion_conditioner/multi_head_attention_1/query/kernel
O:M 2=audio_diffusion_conditioner/multi_head_attention_1/query/bias
T:R� 2=audio_diffusion_conditioner/multi_head_attention_1/key/kernel
M:K 2;audio_diffusion_conditioner/multi_head_attention_1/key/bias
V:T� 2?audio_diffusion_conditioner/multi_head_attention_1/value/kernel
O:M 2=audio_diffusion_conditioner/multi_head_attention_1/value/bias
a:_ �2Jaudio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel
W:U�2Haudio_diffusion_conditioner/multi_head_attention_1/attention_output/bias
>:<
��2*audio_diffusion_conditioner/dense_8/kernel
7:5�2(audio_diffusion_conditioner/dense_8/bias
>:<
��2*audio_diffusion_conditioner/dense_9/kernel
7:5�2(audio_diffusion_conditioner/dense_9/bias
F:D�27audio_diffusion_conditioner/layer_normalization_2/gamma
E:C�26audio_diffusion_conditioner/layer_normalization_2/beta
 "
trackable_list_wrapper
C
0
	1

2
3
4"
trackable_list_wrapper
'
^0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
<__inference_audio_diffusion_conditioner_layer_call_fn_225584input_1"�
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
<__inference_audio_diffusion_conditioner_layer_call_fn_225629input_1"�
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
W__inference_audio_diffusion_conditioner_layer_call_and_return_conditional_losses_225448input_1"�
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
W__inference_audio_diffusion_conditioner_layer_call_and_return_conditional_losses_225539input_1"�
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
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
k_random_generator"
_tf_keras_layer
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

kernel
bias"
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
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_sequential_layer_call_fn_225249
+__inference_sequential_layer_call_fn_225266�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_sequential_layer_call_and_return_conditional_losses_225201
F__inference_sequential_layer_call_and_return_conditional_losses_225232�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_multi_head_attention_1_layer_call_fn_225773
7__inference_multi_head_attention_1_layer_call_fn_225796�
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
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225832
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225868�
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

kernel
bias"
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

kernel
bias"
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

kernel
bias"
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

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_8_layer_call_fn_225877�
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
C__inference_dense_8_layer_call_and_return_conditional_losses_225888�
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
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_9_layer_call_fn_225897�
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
C__inference_dense_9_layer_call_and_return_conditional_losses_225908�
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
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
6__inference_layer_normalization_2_layer_call_fn_225917�
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
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_225939�
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
�
W0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19"
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
$__inference_signature_wrapper_225750input_1"�
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
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_5_layer_call_fn_225948�
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
 z�trace_0
�
�trace_02�
C__inference_dense_5_layer_call_and_return_conditional_losses_225966�
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
 z�trace_0
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
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_2_layer_call_fn_225971
*__inference_dropout_2_layer_call_fn_225976�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_2_layer_call_and_return_conditional_losses_225988
E__inference_dropout_2_layer_call_and_return_conditional_losses_225993�
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
 z�trace_0z�trace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_6_layer_call_fn_226002�
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
 z�trace_0
�
�trace_02�
C__inference_dense_6_layer_call_and_return_conditional_losses_226020�
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
 z�trace_0
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
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_3_layer_call_fn_226025
*__inference_dropout_3_layer_call_fn_226030�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_3_layer_call_and_return_conditional_losses_226042
E__inference_dropout_3_layer_call_and_return_conditional_losses_226047�
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
 z�trace_0z�trace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_7_layer_call_fn_226056�
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
 z�trace_0
�
�trace_02�
C__inference_dense_7_layer_call_and_return_conditional_losses_226066�
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
 z�trace_0
 "
trackable_list_wrapper
C
,0
-1
.2
/3
04"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_sequential_layer_call_fn_225249input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
+__inference_sequential_layer_call_fn_225266input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
F__inference_sequential_layer_call_and_return_conditional_losses_225201input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
F__inference_sequential_layer_call_and_return_conditional_losses_225232input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
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
=0
>1
?2
@3
A4
B5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_multi_head_attention_1_layer_call_fn_225773queryvaluekey"�
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
7__inference_multi_head_attention_1_layer_call_fn_225796queryvaluekey"�
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
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225832queryvaluekey"�
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
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225868queryvaluekey"�
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
0
1"
trackable_list_wrapper
.
0
1"
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
0
1"
trackable_list_wrapper
.
0
1"
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
0
1"
trackable_list_wrapper
.
0
1"
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
0
1"
trackable_list_wrapper
.
0
1"
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
(__inference_dense_8_layer_call_fn_225877inputs"�
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
C__inference_dense_8_layer_call_and_return_conditional_losses_225888inputs"�
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
(__inference_dense_9_layer_call_fn_225897inputs"�
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
C__inference_dense_9_layer_call_and_return_conditional_losses_225908inputs"�
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
6__inference_layer_normalization_2_layer_call_fn_225917inputs"�
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
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_225939inputs"�
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
':%
��2Adam/m/dense_5/kernel
':%
��2Adam/v/dense_5/kernel
 :�2Adam/m/dense_5/bias
 :�2Adam/v/dense_5/bias
':%
��2Adam/m/dense_6/kernel
':%
��2Adam/v/dense_6/kernel
 :�2Adam/m/dense_6/bias
 :�2Adam/v/dense_6/bias
':%
��2Adam/m/dense_7/kernel
':%
��2Adam/v/dense_7/kernel
 :�2Adam/m/dense_7/bias
 :�2Adam/v/dense_7/bias
[:Y� 2FAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/kernel
[:Y� 2FAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/kernel
T:R 2DAdam/m/audio_diffusion_conditioner/multi_head_attention_1/query/bias
T:R 2DAdam/v/audio_diffusion_conditioner/multi_head_attention_1/query/bias
Y:W� 2DAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/kernel
Y:W� 2DAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/kernel
R:P 2BAdam/m/audio_diffusion_conditioner/multi_head_attention_1/key/bias
R:P 2BAdam/v/audio_diffusion_conditioner/multi_head_attention_1/key/bias
[:Y� 2FAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/kernel
[:Y� 2FAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/kernel
T:R 2DAdam/m/audio_diffusion_conditioner/multi_head_attention_1/value/bias
T:R 2DAdam/v/audio_diffusion_conditioner/multi_head_attention_1/value/bias
f:d �2QAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel
f:d �2QAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/kernel
\:Z�2OAdam/m/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias
\:Z�2OAdam/v/audio_diffusion_conditioner/multi_head_attention_1/attention_output/bias
C:A
��21Adam/m/audio_diffusion_conditioner/dense_8/kernel
C:A
��21Adam/v/audio_diffusion_conditioner/dense_8/kernel
<::�2/Adam/m/audio_diffusion_conditioner/dense_8/bias
<::�2/Adam/v/audio_diffusion_conditioner/dense_8/bias
C:A
��21Adam/m/audio_diffusion_conditioner/dense_9/kernel
C:A
��21Adam/v/audio_diffusion_conditioner/dense_9/kernel
<::�2/Adam/m/audio_diffusion_conditioner/dense_9/bias
<::�2/Adam/v/audio_diffusion_conditioner/dense_9/bias
K:I�2>Adam/m/audio_diffusion_conditioner/layer_normalization_2/gamma
K:I�2>Adam/v/audio_diffusion_conditioner/layer_normalization_2/gamma
J:H�2=Adam/m/audio_diffusion_conditioner/layer_normalization_2/beta
J:H�2=Adam/v/audio_diffusion_conditioner/layer_normalization_2/beta
0
�0
�1"
trackable_list_wrapper
.
�	variables"
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
�B�
(__inference_dense_5_layer_call_fn_225948inputs"�
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
C__inference_dense_5_layer_call_and_return_conditional_losses_225966inputs"�
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
*__inference_dropout_2_layer_call_fn_225971inputs"�
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
*__inference_dropout_2_layer_call_fn_225976inputs"�
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
E__inference_dropout_2_layer_call_and_return_conditional_losses_225988inputs"�
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
E__inference_dropout_2_layer_call_and_return_conditional_losses_225993inputs"�
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
(__inference_dense_6_layer_call_fn_226002inputs"�
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
C__inference_dense_6_layer_call_and_return_conditional_losses_226020inputs"�
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
*__inference_dropout_3_layer_call_fn_226025inputs"�
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
*__inference_dropout_3_layer_call_fn_226030inputs"�
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
E__inference_dropout_3_layer_call_and_return_conditional_losses_226042inputs"�
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
E__inference_dropout_3_layer_call_and_return_conditional_losses_226047inputs"�
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
(__inference_dense_7_layer_call_fn_226056inputs"�
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
C__inference_dense_7_layer_call_and_return_conditional_losses_226066inputs"�
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
!__inference__wrapped_model_225110!" 1�.
'�$
"�
input_1����������
� "4�1
/
output_1#� 
output_1����������	�
W__inference_audio_diffusion_conditioner_layer_call_and_return_conditional_losses_225448|!" 5�2
+�(
"�
input_1����������
p
� "-�*
#� 
tensor_0����������	
� �
W__inference_audio_diffusion_conditioner_layer_call_and_return_conditional_losses_225539|!" 5�2
+�(
"�
input_1����������
p 
� "-�*
#� 
tensor_0����������	
� �
<__inference_audio_diffusion_conditioner_layer_call_fn_225584q!" 5�2
+�(
"�
input_1����������
p
� ""�
unknown����������	�
<__inference_audio_diffusion_conditioner_layer_call_fn_225629q!" 5�2
+�(
"�
input_1����������
p 
� ""�
unknown����������	�
C__inference_dense_5_layer_call_and_return_conditional_losses_225966e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_5_layer_call_fn_225948Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_dense_6_layer_call_and_return_conditional_losses_226020e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_6_layer_call_fn_226002Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_dense_7_layer_call_and_return_conditional_losses_226066e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_7_layer_call_fn_226056Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_dense_8_layer_call_and_return_conditional_losses_225888e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_8_layer_call_fn_225877Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_dense_9_layer_call_and_return_conditional_losses_225908e 0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_9_layer_call_fn_225897Z 0�-
&�#
!�
inputs����������
� ""�
unknown�����������
E__inference_dropout_2_layer_call_and_return_conditional_losses_225988e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
E__inference_dropout_2_layer_call_and_return_conditional_losses_225993e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
*__inference_dropout_2_layer_call_fn_225971Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
*__inference_dropout_2_layer_call_fn_225976Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
E__inference_dropout_3_layer_call_and_return_conditional_losses_226042e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
E__inference_dropout_3_layer_call_and_return_conditional_losses_226047e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
*__inference_dropout_3_layer_call_fn_226025Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
*__inference_dropout_3_layer_call_fn_226030Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_225939e!"0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
6__inference_layer_normalization_2_layer_call_fn_225917Z!"0�-
&�#
!�
inputs����������
� ""�
unknown�����������
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225832����
���
$�!
query����������
$�!
value����������
"�
key����������

 
p 
p
p 
� "1�.
'�$
tensor_0����������
� �
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_225868����
���
$�!
query����������
$�!
value����������
"�
key����������

 
p 
p 
p 
� "1�.
'�$
tensor_0����������
� �
7__inference_multi_head_attention_1_layer_call_fn_225773����
���
$�!
query����������
$�!
value����������
"�
key����������

 
p 
p
p 
� "&�#
unknown�����������
7__inference_multi_head_attention_1_layer_call_fn_225796����
���
$�!
query����������
$�!
value����������
"�
key����������

 
p 
p 
p 
� "&�#
unknown�����������
F__inference_sequential_layer_call_and_return_conditional_losses_225201r9�6
/�,
"�
input_1����������
p

 
� "-�*
#� 
tensor_0����������
� �
F__inference_sequential_layer_call_and_return_conditional_losses_225232r9�6
/�,
"�
input_1����������
p 

 
� "-�*
#� 
tensor_0����������
� �
+__inference_sequential_layer_call_fn_225249g9�6
/�,
"�
input_1����������
p

 
� ""�
unknown�����������
+__inference_sequential_layer_call_fn_225266g9�6
/�,
"�
input_1����������
p 

 
� ""�
unknown�����������
$__inference_signature_wrapper_225750�!" <�9
� 
2�/
-
input_1"�
input_1����������"4�1
/
output_1#� 
output_1����������	