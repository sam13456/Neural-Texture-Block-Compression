import math

def calc_grid_sizes(num_levels, base_res, finest_res, feature_dim, bits_list):
    if num_levels == 1:
        resolutions = [base_res]
    else:
        b = math.exp((math.log(finest_res) - math.log(base_res)) / (num_levels - 1))
        resolutions = [int(math.floor(base_res * (b ** l) + 1e-9)) for l in range(num_levels)]
        resolutions[-1] = finest_res
    
    print(f"  {'Level':>5} {'Res':>6} {'Params':>10} {'Bits':>4} {'Bytes(packed)':>14} {'KB':>8}")
    print("  " + "-" * 55)
    total_packed = 0
    for i, r in enumerate(resolutions):
        params = r * r * feature_dim
        bits = bits_list[i]
        if bits in [1, 2, 4]:
            packed_bytes = (params * bits + 7) // 8
        else:
            packed_bytes = params
        total_packed += packed_bytes
        print(f"  {i:>5} {r:>6} {params:>10} {bits:>4} {packed_bytes:>14} {packed_bytes/1024:>7.1f}")
    print("  " + "-" * 55)
    print(f"  TOTAL: {total_packed:,} bytes = {total_packed/1024:.1f} KB = {total_packed/1024/1024:.2f} MB")
    print()

print("=== ENDPOINT NETWORK (7 levels, base=16, finest=1024, feat_dim=2) ===")
print("--- Config: [8,8,8,8,8,8,8] ---")
calc_grid_sizes(7, 16, 1024, 2, [8,8,8,8,8,8,8])

print("=== COLOR NETWORK (8 levels, base=16, finest=2048, feat_dim=2) ===")
print("--- NEW config: [8,4,4,4,4,4,4,4] ---")
calc_grid_sizes(8, 16, 2048, 2, [8,2,2,2,2,2,4,4])

print("--- OLD config: [8,8,8,8,4,4,4,4] ---")
calc_grid_sizes(8, 16, 2048, 2, [8,8,8,8,4,4,4,4])

print("--- All 8-bit: [8,8,8,8,8,8,8,8] ---")
calc_grid_sizes(8, 16, 2048, 2, [8,8,8,8,8,8,8,8])
