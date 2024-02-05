mple_outputs):
        sample_output_dec = tokenizer.decode(sample_output, skip_special_tokens=True)
        print(f"{i}: {sample_output_dec}")