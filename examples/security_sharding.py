from pyfolds.serialization.sharding_raid import RAIDSharding

r = RAIDSharding()
shards = r.split(b"hello" * 100)
rec = r.reconstruct([s for i, s in enumerate(shards) if i != 1], [0,2,3,4])
print(len(rec))
