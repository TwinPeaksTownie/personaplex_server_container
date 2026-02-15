import { useState, useEffect, useCallback } from "react";

export function useVoices() {
  const [voices, setVoices] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);

  const fetchVoices = useCallback(() => {
    fetch("/api/voices")
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data.voices)) {
          setVoices(data.voices.sort());
        }
      })
      .catch((err) => console.error("Failed to fetch voices:", err));
  }, []);

  useEffect(() => {
    fetchVoices();
  }, [fetchVoices]);

  const uploadVoice = useCallback(async (file: File): Promise<string | null> => {
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file, file.name);
      const res = await fetch("/api/voices/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        console.error("Upload failed:", data.error);
        return null;
      }
      fetchVoices();
      return data.filename as string;
    } catch (err) {
      console.error("Upload failed:", err);
      return null;
    } finally {
      setUploading(false);
    }
  }, [fetchVoices]);

  return { voices, uploadVoice, uploading, refreshVoices: fetchVoices };
}
