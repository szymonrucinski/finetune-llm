st of values to replace Q4_0
values=("Q4_0" "Q4_1" "Q5_0" "Q5_1" "Q2_K" "Q3_K" "Q3_K_S" "Q3_K_M" "Q3_K_L" "Q4_K" "Q4_K_S" "Q4_K_K" "Q5_K" "Q5_K_S" "Q5_K_M" "Q6_K" "Q8_0")

# The original command
original_command="./quantize /finetune-llm-main/krakowiak-v2-7b-gguf/krakowiak-v2-7b-gguf.fp16.bin /finetune-llm-main/krakowiak-v2-7b-gguf/krakowiak-v2-7b-gguf.Q4_0.bin Q4_0"

# Loop over the list and replace Q4_0 with each value
for value in "${values[@]}"; do
	    command_with_replacement="./quantize /finetune-llm-main/krakowiak-v2-7b-gguf/krakowiak-v2-7b-gguf.fp16.bin /finetune-llm-main/krakowiak-v2-7b-gguf/krakowiak-v2-7b-gguf.$value.bin $value"
	        echo "Executing: $command_with_replacement"
		    # Uncomment the next line to actually run the command
		        eval "$command_with_replacement"
		done

