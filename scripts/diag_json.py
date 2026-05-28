
import json, os
qa_dir = os.environ.get("QA_DIR")
import glob
orig = sorted(glob.glob(os.path.join(qa_dir, 'v2v4real_3d_grounding_qa_dataset_nq*sm*.json')))
orig = [f for f in orig if not f.endswith('_satdesc.json') and not f.endswith('.orig_satdesc')]
print("Found base files:", len(orig))
for o in orig[:1]:
    sat = o.replace(".json", "_satdesc.json")
    if not os.path.exists(sat):
        print(f"NO SATDESC FOR {os.path.basename(o)}"); continue
    a = json.load(open(o))
    b = json.load(open(sat))
    print(f"\n{os.path.basename(o)}")
    print(f"  orig entries:    {len(a)}")
    print(f"  satdesc entries: {len(b)}")
    print(f"  orig conv[0][:200]:    {a[0]['conversations'][0]['value'][:200]!r}")
    print(f"  satdesc conv[0][:500]: {b[0]['conversations'][0]['value'][:500]!r}")
    # Find any field added in satdesc that isn't in orig
    new_keys = set(b[0].keys()) - set(a[0].keys())
    print(f"  new keys in satdesc: {new_keys}")