# from app.pdf.ingest_pdf import process_all_sops,process_pdf_to_qdrant
from app.pdf.ingest_pdf import process_all_sops
from app.trainings.ingest_tps import process_all_trainings
from app.forms.ingest_forms import process_all_forms
from app.tasks.ingest_tasks import process_all_tasks
from app.audits.ingest_audits import process_all_audits
from app.guides.ingest_guide import process_individual_guides

def main():
    """Process all document types sequentially"""
    
    # Define the processing functions and their names
    processors = [
        # ("Trainings", process_all_trainings),
        # ("Forms", process_all_forms),
        # ("Tasks", process_all_tasks),
        # ("Audits", process_all_audits),
        # ("Guides",process_individual_guides),
        # ("SOPs", process_all_sops),
    ]
    
    print("🚀 Starting document processing pipeline...")
    print("=" * 50)
    
    for doc_type, processor_func in processors:
        try:
            print(f"\n📋 Processing {doc_type}...")
            print("-" * 30)
            processor_func()
            print(f"✅ Completed processing {doc_type}")
        except Exception as e:
            print(f"❌ Failed to process {doc_type}: {e}")
            # Continue with next processor even if one fails
            continue
    
    print("\n🎉 Document processing pipeline completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()