#!/bin/bash

dir=$1

datatypes="i8 u8 i32 u32 i64 u64 f16 f32 f64"
if [ "$CUDA_MAJOR" -ge 11 ]
then
    datatypes+=" bf16"
fi

targets="GENOBJS := \\\\\n"

for base in {"all_gather","all_reduce"}; do
  opn=0
  for op in {"sum","prod"}; do
    if [ "$base" = "all_gather" ] && [ "$op" != "sum" ]; then
      continue
    fi
    dtn=0
    # Order must match that of the ncclDataType_t enum
    for dt in ${datatypes}; do
      # Generate a unique filename for each compilation unit,
      # otherwise the __nv_module_id may conflict at link time
      echo "${dir}/${base}_${op}_${dt}.cu : src/${base}.cu"
      echo "	@printf \"Copying    %-35s > %s\\\\n\" \$< \$@"
      echo "	cp \$< \$@"
      echo ""
      # Compile the file
      echo "${dir}/${base}_${op}_${dt}.o : ${dir}/${base}_${op}_${dt}.cu src/${base}.cu ${dir}/${base}.dep"

      echo "	@printf \"Compiling  %-35s > %s\\\\n\" src/${base}.cu ${dir}/${base}_${op}_${dt}.o"
      echo "	mkdir -p ${dir}"
      echo "	\${NVCC} -DMCCS_OP=${opn} -DMCCS_TYPE=${dtn} \${NVCUFLAGS} -dc \$< -o \$@"
      echo ""
      targets="$targets\t${dir}/${base}_${op}_${dt}.o \\\\\n"
      dtn=$(($dtn + 1))
    done
    opn=$(($opn + 1))
  done
done
echo -e "$targets"
